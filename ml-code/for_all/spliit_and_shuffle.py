from input_data import *
import numpy as np
import random

' divide the dataset to workers datasets'


def get_workers_dataset(dataset, num_of_workers, batchsize):
    batch_size = batchsize

    # loading the dataset
    number_of_examples = len(dataset.train.images)

    # getting the batches for each atrribute : images , labels
    mnist_images_batches = [dataset.train.images[i * batch_size:(i + 1) * batch_size] for i in
                            range(0, int(np.floor(number_of_examples / batch_size)))]
    mnist_labels_batches = [dataset.train.labels[i * batch_size:(i + 1) * batch_size] for i in
                            range(0, int(np.floor(number_of_examples / batch_size)))]

    # creating the ( image , label ) batches of the data
    train_feed = [(mnist_images_batches[i], mnist_labels_batches[i]) for i in
                  range(0, int(number_of_examples / batch_size))]

    # shuffeling our batches so each worker will have different parts of the dataset
    np.random.shuffle(train_feed)

    # getting the dataset for each worker as an array of tuple batches : ( image_batch , label_batch )
    train_feed_workers_dataset = [train_feed[int(np.floor(number_of_examples / (batch_size * num_of_workers))) * i: int(
        np.floor(number_of_examples / (batch_size * num_of_workers))) * (i + 1)] for i in range(0, num_of_workers)]
    return train_feed_workers_dataset

# Dataset = read_data_sets('MNIST_data', one_hot=True)
# worker_datasets = get_workers_dataset(Dataset, 1, 1000)

# Here we will get all the batches of ( image , label ) of worker_a
# worker_a = worker_datasets[0]

# Here we will get the first batch of images out of worker_a dataset
# print(" the first batch of images out of worker_a dataset is ", (worker_a[0][0]))
