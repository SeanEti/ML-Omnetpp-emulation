"""
Distributed Tensorflow 0.8.0 example of using data parallelism and share model parameters.
Trains a simple sigmoid neural network on mnist for 20 epochs on three machines using one parameter server.
Change the hardcoded host urls below with your own hosts.
Run like this:
pc-01$ python example.py --job_name="ps" --task_index=0
pc-02$ python example.py --job_name="worker" --task_index=0
pc-03$ python example.py --job_name="worker" --task_index=1
pc-04$ python example.py --job_name="worker" --task_index=2
More details here: ischlag.github.io
"""

from __future__ import print_function

from spliit_and_shuffle import *
from input_data import *
import tensorflow as tf
# import sys
import time

parameter_servers = ["192.168.4.21:4000"]
workers = ["192.168.2.21:4545", "192.168.3.21:4444", "192.168.5.21:4646", "192.168.6.21:4747"]

cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

# input flags
tf.flags.DEFINE_string("job_name", "worker", "Either 'ps' or 'worker'")
tf.flags.DEFINE_integer("task_index", 3, "Index of task within the job")
tf.flags.DEFINE_boolean(
    "sync_replicas", False,
    "Use the sync_replicas (synchronized replicas) mode, "
    "wherein the parameter updates from workers are aggregated "
    "before applied to avoid stale gradients")
FLAGS = tf.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)

# config
batch_size = 100
learning_rate = 0.0001
training_epochs = 5
# logs_path = "/tmp/mnist/1"

# load mnist data set
mnist = read_data_sets('MNIST_data', one_hot=True)

# get specific dataset
worker_dataset = get_workers_dataset(mnist, len(workers), batchsize=batch_size)[FLAGS.task_index]

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    is_chief = (FLAGS.task_index == 0)
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):

        # count the number of updates
        global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        # input images
        with tf.name_scope('input'):
            # None -> batch size can be any size, 784 -> flattened mnist image
            x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
            # target 10 output classes
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

        # model parameters will change during training so we use tf.Variable
        tf.set_random_seed(1)
        with tf.name_scope("weights"):
            W1 = tf.Variable(tf.random_normal([784, 100]))
            W2 = tf.Variable(tf.random_normal([100, 10]))

        # bias
        with tf.name_scope("biases"):
            b1 = tf.Variable(tf.zeros([100]))
            b2 = tf.Variable(tf.zeros([10]))

        # implement model
        with tf.name_scope("softmax"):
            # y is our prediction
            z2 = tf.add(tf.matmul(x, W1), b1)
            a2 = tf.nn.sigmoid(z2)
            z3 = tf.add(tf.matmul(a2, W2), b2)
            y = tf.nn.softmax(z3)

        # specify cost function
        with tf.name_scope('cross_entropy'):
            # this is our cost
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))

        # specify optimizer
        with tf.name_scope('train'):
            # optimizer is an "operation" which we can execute in a session
            grad_op = tf.train.GradientDescentOptimizer(learning_rate)

            # create an optimizer then wrap it with SynceReplicasOptimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            optimizer1 = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=len(workers),
                                                        total_num_replicas=len(workers))
            opt = optimizer1.minimize(cross_entropy, global_step=global_step)  # averages gradients
            # opt = optimizer1.minimize(REPLICAS_TO_AGGREGATE*loss,
            #                           global_step=global_step) # hackily sums gradients

        # train_op = grad_op.minimize(cross_entropy, global_step=global_step)
        # init_token_op = rep_op.get_init_tokens_op()
        # chief_queue_runner = rep_op.get_chief_queue_runner()

        with tf.name_scope('Accuracy'):
            # accuracy
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # create a summary for our cost and accuracy
        tf.summary.scalar("cost", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)

        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

    # sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
    #                         global_step=global_step,
    #                         init_op=init_op, recovery_wait_secs=1)

    # Session
    sync_replicas_hook = optimizer1.make_session_run_hook(is_chief, num_tokens=0)
    stop_hook = tf.train.StopAtStepHook(last_step=training_epochs * len(worker_dataset) + 1)
    hooks = [sync_replicas_hook, stop_hook]

    # Monitored Training Session
    sess = tf.train.MonitoredTrainingSession(master=server.target,
                                             is_chief=is_chief,
                                             hooks=hooks,
                                             stop_grace_period_secs=20)

    print('Starting training on worker %d' % FLAGS.task_index)

    begin_time = time.time()
    frequency = 2
    counter_after_sync = 0

    while not sess.should_stop() and counter_after_sync < training_epochs * len(worker_dataset):
        # is chief
        # if FLAGS.task_index == 0 and FLAGS.sync_replicas:
        #    sv.start_queue_runners(sess, [chief_queue_runner])
        #    sess.run(init_token_op)

        # create log writer object (this will log on every machine)
        #        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # perform training cycles
        start_time = time.time()
        for epoch in range(training_epochs):
            # number of batches in one epoch
            batch_count = len(worker_dataset)
            count = 0
            for i in range(batch_count):
                batch_x, batch_y = worker_dataset[i][0], worker_dataset[i][1]

                # perform the operations we defined earlier on batch
                _, cost, summary, step = sess.run(
                    [opt, cross_entropy, summary_op, global_step],
                    feed_dict={x: batch_x, y_: batch_y})
                # if is_chief: time.sleep(1)
                # time.sleep(1)
                #                writer.add_summary(summary, step)

                counter_after_sync += 1
                count += 1
                if count % frequency == 0 or i + 1 == batch_count:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print("worker : %1d " % FLAGS.task_index,
                          "Step: %d," % (step + 1),
                          " Epoch: %2d," % (epoch + 1),
                          " Batch: %3d of %3d," % (i + 1, batch_count),
                          " Cost: %.4f," % cost)
                    count = 0

    val_xent = sess.run(cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("After %d training step(s), validation cross entropy = %g" % (step, val_xent))
    print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    print("Total Time: %3.2fs" % float(time.time() - begin_time))
    print("Final Cost: %.4f" % cost)
    print('Done', FLAGS.task_index)
    time.sleep(10)  # grace period to wait before closing session
    sess.close()
    print('Session from worker %d closed cleanly' % FLAGS.task_index)
