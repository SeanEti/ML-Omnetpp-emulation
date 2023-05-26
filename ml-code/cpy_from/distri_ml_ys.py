import collections
import operator
# import os
import sys
import torch
# import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data import DataLoader
import argparse
import functools
import socket
# import time
import pickle


# ===================================================
#                       CLASSES
# ===================================================

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# ===================================================
#                     FUNCTIONS
# ===================================================

def split_dataset(train_loader_data, num_of_workers):
    # Get the total number of samples in the dataset
    dataset_size = len(train_loader_data.dataset)

    # Calculate the size of each worker's subset
    subset_size = dataset_size // num_of_workers

    # Dynamically split the dataset among workers
    worker_datasets = torch.utils.data.random_split(train_loader_data.dataset, [subset_size] * num_of_workers)

    return worker_datasets


def train(epoch_num, net, func_optim, train_loader_data, log_freq=10):
    """
    train model
    """
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader_data):
        func_optim.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        func_optim.step()

        if batch_idx % log_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_num, batch_idx * len(data), len(train_loader_data.dataset),
                100. * batch_idx / len(train_loader_data), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch_num-1)*len(train_loader_data.dataset)))
            torch.save(net.state_dict(), 'model.pth')
            torch.save(func_optim.state_dict(), 'optimizer.pth')
    return net.state_dict()


def test(net, test_loader_data):
    """
    test model
    """
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader_data:
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader_data.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader_data.dataset),
        100. * correct / len(test_loader_data.dataset)))


def get_new_state(worker_model_dic):
    """
    gets a dictionary of ('worker address': 'his state dict') and returns a new state dict where each value is the
    average value of the given states.
    """
    dicts = list(worker_model_dic.values())
    new_state = dict(functools.reduce(operator.add,
                                      map(collections.Counter, dicts)))
    new_state.update(
        (key, value / len(dicts)) for key, value in dicts.items()
    )
    return new_state


def broadcast_state_dict(new_state, workers_addrs, server_sock):
    """
    Sending the new state dictionary of the model to all the workers
    """
    data_string = pickle.dumps(new_state)
    for addr in workers_addrs:
        if server_sock.sendto(data_string, addr) == -1:
            print(f"ERR: Failed to send new state of model to {addr} while broadcasting...")
    pass


# ===================================================
#                       MAIN
# ===================================================

parser = argparse.ArgumentParser(description="arguments for project",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-j", "--job_name", help="Type of job of this worker (ps or worker)")
parser.add_argument("-t", "--task_index", type=int, default=0, help="Number of worker(ps gets 0)")
parser.add_argument("-a", "--addresses", help="path to csv file with address for workers")
parser.add_argument("-m", "--model", help="The ML model to implement")
parser.add_argument("-s", "--server", help="ip and port of parameter server")
args = vars(parser.parse_args())

# save IPs and ports of all workers and parameter-server from arguments
with open(args["addresses"], 'r') as fd:
    workers = fd.read().splitlines()[0].split(',')
num_of_workers = len(workers)
parameter_server = args["server"]
print(f'Received IPs:\nWorkers: {workers}\nParameter Server: {parameter_server}')

# setting up
num_of_epochs = 1
batch_size_train = 64
learning_rate = 0.01
momentum = 0.5
log_interval = 10
batch_size_test = 1000

torch.backends.cudnn.enabled = False
torch.manual_seed(696969)

serv_ip, serv_port = parameter_server.split(':')    # get IP and port of parameter-server
serv_port = int(serv_port)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

if args["job_name"] == 'worker':
    print(f'This is worker{args["task_index"]}')

    # get data
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=batch_size_train, shuffle=True)

    train_loader = split_dataset(train_loader, num_of_workers)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size_test, shuffle=True)

    # initialize model to train:
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    # track progress init
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(num_of_epochs + 1)]

    # start to train model
    print("Starting training...")
    for epoch in range(1, num_of_epochs + 1):
        curr_state = train(epoch, network, optimizer, train_loader, log_interval)
        # TODO send gradients to server
        state_string = pickle.dumps(curr_state, -1)
        print(f"Size of state dict is {sys.getsizeof(state_string)}B")

        if sock.sendto(state_string, (serv_ip, serv_port)) == -1:
            print("ERR: Failed to send current state of model in worker :(...")
        print("sent state to server...")
        # TODO wait to receive new params from server
        message = sock.recvfrom(2048)
        new_state_dict = pickle.loads(message[0])

        # TODO update model to new params
        network.load_state_dict(new_state_dict)
        print(f"Received new state dictionary from server:\n{new_state_dict}")

elif args["job_name"] == 'ps':
    print("This is a parameter server!")
    # bind socket
    sock.bind((serv_ip, serv_port))

    num_of_workers_who_shared_model = 0
    worker_model_dict = {}

    # listen for incoming datagrams
    print("Listening...")
    while True:
        bytes_and_address = sock.recvfrom(2048)
        message = pickle.loads(bytes_and_address[0])    # deserialize model state dict
        address = bytes_and_address[1]                  # save address of worker

        worker_model_dict[address] = message            # save received model in dictionary
        num_of_workers_who_shared_model += 1
        if num_of_workers_who_shared_model == num_of_workers:
            new_state_dict = get_new_state(worker_model_dict)
            broadcast_state_dict(new_state_dict, len(worker_model_dict.keys()))
            # reset counters and wait for another round of new models
            num_of_workers_who_shared_model = 0
            worker_model_dict = {}
else:
    print("ERR: Wrong job entered in argument...")
    exit(-1)
