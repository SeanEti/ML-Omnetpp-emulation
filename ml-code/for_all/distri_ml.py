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
import tensorflow as tf
from spliit_and_shuffle import *
from input_data import *
from CNN_implementation import *
from functools import reduce
# import sys
import time
import argparse

parser = argparse.ArgumentParser(description="arguments for project",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-j", "--job_name", help="Type of job of this worker (ps or worker)")
parser.add_argument("-t", "--task_index", type=int, default=0, help="Number of worker(ps gets 0)")
parser.add_argument("-a", "--addresses", help="path to csv file with address for workers")
parser.add_argument("-m", "--model", help="The ML model to implement")
parser.add_argument("-s", "--server", help="ip and port of parameter server")


args = vars(parser.parse_args())

with open(args["addresses"], 'r') as fd:
    workers = fd.read().splitlines()[0].split(',')

parameter_servers = [args["server"]]

print(f'Received IPs:\nWorkers: {workers}\nParameter Server: {parameter_servers}')

cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

# input flags
#tf.flags.DEFINE_string("job_name", args["job_name"],  "Either 'ps' or 'worker'")
#tf.flags.DEFINE_integer("task_index", args["task_index"], "Index of task within the job")
#tf.flags.DEFINE_boolean(
#    "sync_replicas", False,
#    "Use the sync_replicas (synchronized replicas) mode, "
#    "wherein the parameter updates from workers are aggregated "
#    "before applied to avoid stale gradients")
#tf.flags.DEFINE_string("model", "SGD", "could be: SGD , Softmax regression, "
#                                    "Multi-layer_perceptron, CNN")
#FLAGS = tf.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster,
                         job_name= args["job_name"],
                         task_index=args["task_index"])

# config
batch_size = 100
learning_rate = 0.1
dropout_keep_proba = 0.5
training_epochs = 10
number_of_workers = len(workers)
# logs_path = "/tmp/mnist/1"

# Architecture
n_hidden_1 = 128
n_hidden_2 = 256
n_features = 784
n_input = n_features
n_classes = 10
image_width, image_height = 28, 28

print("Reading Data...")
# load mnist data set
mnist = read_data_sets('MNIST_data', one_hot=True)

# get specific dataset
worker_dataset = get_workers_dataset(mnist, num_of_workers=number_of_workers, batchsize=batch_size)[args["task_index"]]

if args["job_name"] == "ps":
    print("Server joining...")
    server.join()
elif args["job_name"] == "worker":
    print("This is a worker!")
    is_chief = (args["task_index"] == 0)
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % args["task_index"],
            cluster=cluster)):

        # count the number of updates
        global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        

        # input images
        with tf.name_scope('input'):
            # None -> batch size can be any size, 784 -> flattened mnist image
            x = tf.placeholder(tf.float32, shape=[None, n_features], name="x-input")
            # target 10 output classes
            y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="y-input")

            if args["model"] == "CNN":
                keep_proba = tf.placeholder(tf.float32, shape=None, name='keep_proba')

        # model parameters will change during training so we use tf.Variable
        tf.set_random_seed(1)
        with tf.name_scope("weights"):
            if args["model"] == "SGD":
                W1 = tf.Variable(tf.random_normal([784, 100]))
                W2 = tf.Variable(tf.random_normal([100, 10]))

            if args["model"] == "Softmax_regression":
                # Model parameters
                W1 = tf.Variable(tf.zeros(shape=[n_features, n_classes], dtype=tf.float32))

        # bias
        with tf.name_scope("biases"):
            if args["model"] == "SGD":
                b1 = tf.Variable(tf.zeros([100]))
                b2 = tf.Variable(tf.zeros([10]))
            if args["model"] == "Softmax_regression":
                b1 = tf.Variable([[n_classes]], dtype=tf.float32)

        # implement model
        with tf.name_scope("model"):
            if args["model"] == "SGD":
                print("Chosen SGD!")
                # y is our prediction
                z2 = tf.add(tf.matmul(x, W1), b1)
                a2 = tf.nn.sigmoid(z2)
                z3 = tf.add(tf.matmul(a2, W2), b2)
                y = tf.nn.softmax(z3)

            if args["model"] == "Softmax_regression":
                linear = tf.matmul(x, W1) + b1
                pred_proba = tf.nn.softmax(linear, name='predict_probas')

            if args["model"] == "Multi-layer_perceptron":
                # Multilayer perceptron
                layer_1 = tf.layers.dense(x, n_hidden_1, activation=tf.nn.relu,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
                layer_2 = tf.layers.dense(layer_1, n_hidden_2, activation=tf.nn.relu,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
                out_layer = tf.layers.dense(layer_2, n_classes, activation=None)

            if args["model"] == "CNN":
                # Convolutional Neural Network:
                # 2 convolutional layers with maxpool and ReLU activation
                input_layer = tf.reshape(x, shape=[-1, image_width, image_height, 1])

                conv1 = conv2d(input_tensor=input_layer,
                               output_channels=8,
                               kernel_size=(3, 3),
                               strides=(1, 1, 1, 1),
                               activation=tf.nn.relu,
                               name='conv1')

                pool1 = tf.nn.max_pool(conv1,
                                       ksize=(1, 2, 2, 1),
                                       strides=(1, 2, 2, 1),
                                       padding='SAME',
                                       name='maxpool1')

                conv2 = conv2d(input_tensor=pool1,
                               output_channels=16,
                               kernel_size=(3, 3),
                               strides=(1, 1, 1, 1),
                               activation=tf.nn.relu,
                               name='conv2')

                pool2 = tf.nn.max_pool(conv2,
                                       ksize=(1, 2, 2, 1),
                                       strides=(1, 2, 2, 1),
                                       padding='SAME',
                                       name='maxpool2')

                dims = pool2.get_shape().as_list()[1:]
                dims = reduce(lambda x, y: x * y, dims, 1)
                flat = tf.reshape(pool2, shape=(-1, dims))

                out_layer = fully_connected(flat, n_classes, activation=None,
                                            name='logits')

        # specify cost function
        with tf.name_scope('cross_entropy'):
            if args["model"] == "SGD":
                # this is our cost
                cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))

            if args["model"] == "Softmax_regression":
                # cross entropy also known as cost
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=linear, labels=y_))

            if args["model"] == "Multi-layer_perceptron":
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y_)
                cross_entropy = tf.reduce_mean(loss, name='cost')

            if args["model"] == "CNN":
                # Loss and optimizer
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_layer, labels=y_)
                cross_entropy = tf.reduce_mean(loss, name='cost')
            # actual gradients
            # var_grad = tf.gradients(cross_entropy, [y])[0]

        # specify optimizer
        with tf.name_scope('train'):
            # optimizer is an "operation" which we can execute in a session
            grad_op = tf.train.GradientDescentOptimizer(learning_rate)

            # create an optimizer then wrap it with SynceReplicasOptimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            optimizer1 = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=number_of_workers,
                                                        total_num_replicas=number_of_workers)
            opt = optimizer1.minimize(cross_entropy, global_step=global_step)  # averages gradients
            # opt = optimizer1.minimize(REPLICAS_TO_AGGREGATE*loss,
            #                           global_step=global_step) # hackily sums gradients

        # train_op = grad_op.minimize(cross_entropy, global_step=global_step)
        # init_token_op = rep_op.get_init_tokens_op()
        # chief_queue_runner = rep_op.get_chief_queue_runner()

        with tf.name_scope('Accuracy'):
            # accuracy

            if args["model"] == "SGD":
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            if args["model"] == "Softmax_regression":
                # Class prediction
                pred_labels = tf.argmax(pred_proba, 1, name='predict_labels')
                correct_prediction = tf.equal(tf.argmax(y_, 1), pred_labels)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

            if args["model"] == "Multi-layer_perceptron":
                # Prediction
                correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(out_layer, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

            if args["model"] == "CNN":
                # Prediction
                correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(out_layer, 1), name='correct_prediction')
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        # create a summary for our cost and accuracy
        tf.summary.scalar("cost", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)

        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

    run_all = False
    
    # Session
    sync_replicas_hook = optimizer1.make_session_run_hook(is_chief, num_tokens=0)
    stop_hook = tf.train.StopAtStepHook(last_step=training_epochs * len(worker_dataset) + 1)
    hooks = [sync_replicas_hook, stop_hook]

    # Monitored Training Session
    print("Trying to start session with server...")
    sess = tf.train.MonitoredTrainingSession(master=server.target,
                                             is_chief=is_chief,
                                             hooks=hooks,
                                             stop_grace_period_secs=20)

    print('Starting training on worker %d' % args["task_index"])

    begin_time = time.time()
    frequency = 20
    counter_after_sync = 0

    while not sess.should_stop() and counter_after_sync < training_epochs * len(worker_dataset):
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
                    print("worker : %1d " % args["task_index"],
                          "Step: %d," % (step + 1),
                          " Epoch: %2d," % (epoch + 1),
                          " Batch: %3d of %3d," % (i + 1, batch_count),
                          " Cost: %.4f," % cost,
                          " Elapsed time: %3.2fs" % elapsed_time)
                    count = 0

    # val_xent = sess.run(cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    # print("After %d training step(s), validation cross entropy = %g" % (step, val_xent))
    print("The model chosen is ", args["model"])
    print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    print("Total Time: %3.2fs" % float(time.time() - begin_time))
    # print("Final Cost: %.4f" % cost)
    print('Done', args["task_index"])
    time.sleep(10)  # grace period to wait before closing session
    sess.close()
    print('Session from worker %d closed cleanly' % args["task_index"])
