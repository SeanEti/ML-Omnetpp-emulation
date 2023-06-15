#
#   Code by Sean Etinger and Yahel Sar-Shalom
#
# ===================================================
#                       IMPORTS
# ===================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import socket
import time
import collections
import numpy as np
import glob
import os
import scapy.all as scapy


# ===================================================
#                       CLASSES
# ===================================================

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnn_model = nn.Sequential(nn.Conv2d(1, 6, 5), 
                                       nn.Tanh(), 
                                       nn.AvgPool2d(2, stride=2), 
                                       nn.Conv2d(6, 16, 5),
                                       nn.Tanh(), 
                                       nn.AvgPool2d(2, stride=2))

        self.fc_model = nn.Sequential(nn.Linear(256, 120), 
                                      nn.Tanh(), 
                                      nn.Linear(120, 84), 
                                      nn.Tanh(), 
                                      nn.Linear(84, 10))

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x


# ===================================================
#                     FUNCTIONS
# ===================================================

def split_dataset(data, num_of_workers):
    # Get the total number of samples in the dataset
    dataset_size = len(data)

    # Calculate the size of each worker's subset
    subset_size = dataset_size // num_of_workers

    # Dynamically split the dataset among workers
    worker_datasets = torch.utils.data.random_split(data, [subset_size] * num_of_workers)

    return worker_datasets


def recvall(sock):
    """
    Receives and returns a large fragmented message
    """
    BUFF_SIZE = 1448
    data = []
    while True:
        part = sock.recv(BUFF_SIZE)
        data.append(part)
        if len(part) < BUFF_SIZE:
            break
    return b"".join(data)


def train(epoch_num, net, func_optim, func_loss, train_loader_data, train_counter, loss_per_epoch, log_freq=10):
    """
    train model
    """
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader_data):
        func_optim.zero_grad()
        output = net(data)
        loss = func_loss(output, target)
        loss.backward()
        func_optim.step()

        if batch_idx % log_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_num, batch_idx * len(data), len(train_loader_data.dataset),
                100. * batch_idx / len(train_loader_data), loss.item()))
            train_counter.append((batch_idx*64) + ((epoch_num-1)*len(train_loader_data.dataset)))
    loss_per_epoch.append(loss.item())
    return net.state_dict(), train_counter, loss_per_epoch


def test(net, test_loader_data):
    """
    test model
    """
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader_data:
            output = net(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader_data.dataset),
        100. * correct / len(test_loader_data.dataset)))
    return 100. * correct / len(test_loader_data.dataset)


def get_new_state(worker_model_dic, num_of_workers):
    """
    gets list of received state dictionaries and returns a new state dict where each value is the
    average value of the given states.
    """
    dicts = worker_model_dic
    model0 = LeNet()
    model0.load_state_dict(dicts[0], strict=False)

    for idx, state in enumerate(dicts):
        if idx == 0: continue
        model = LeNet()
        model.load_state_dict(state, strict=False)
        for param, param0 in zip(model.parameters(), model0.parameters()):
            param0.data.copy_(param.data + param0.data)

    for param in model0.parameters():
        param.data.copy_(param.data / num_of_workers)

    return model0.state_dict()


def get_average_of_list(lst):
    return sum(lst) / len(lst)


def flatten_list(tensor):
    """
    Take evert multimentinal tensor and flatten it to a one dimentional 
    """
    flattened_list = []
    for item in tensor:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


def writeFileInFormat(statedict, filename):
    """
    This function wirtes the statedict in a specified format in filename file
    In general for each parameter:
    name <name-of-parameter>
    <Parameter-value>
    """
    with open(filename, 'w', encoding='utf-8') as my_file:
        for (key, value) in statedict.items():
            #    writing the name of param in a new line
            my_file.write('name ' + key + '\n')
            #    writing the flatten tensor (1D) of every parameter
            for item in torch.flatten(value).tolist():
                my_file.write(f'{item} ')
            my_file.write("\n")
            # my_file.write(str(flatten_list(value)).strip('[,]') + '\n')


def dim(a):
    """
    get the dimentions of every multi-dimensional list
    """
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])


def getDimslist(statedict):
    return [dim(i.tolist()) for i in list(statedict.values())]


def readFormatFile(filename, dims_list):
    """
    This function takes the file and reading it to a state dict
    """
    lines_indices_of_param_name = []
    data_dict = {}
    with open(filename) as f:
        #   Get the rows indices which start with name
        lines = f.readlines()

        # find the lines which has the parameter and put them in a list
        for index, line in enumerate(lines):
            if line.startswith('name'):
                lines_indices_of_param_name.append(index)

        # print(indices_of_param_line)
        for i in range(len(lines_indices_of_param_name) - 1):
            key = lines[lines_indices_of_param_name[i]]
            # taking the specific data for the parameter
            value = (lines[lines_indices_of_param_name[i] + 1:lines_indices_of_param_name[i + 1]])
            value = [s.rstrip("\n") for s in value]
            # Splitting the string and converting to a list of numbers
            numbers = [float(num) for num in value[0].split()]
            data_dict.update({key.strip('\n name'): numbers})
        # for last param
        key = lines[lines_indices_of_param_name[-1]]
        value = (lines[lines_indices_of_param_name[-1] + 1::])
        value = [s.rstrip("\n") for s in value]
        # Splitting the string and converting to a list of numbers
        numbers = [float(num) for num in value[0].split()]
        data_dict.update({key.strip('\n name'): numbers})

    # Converting the lists back to tensors withtheir respected sizes
    for i in data_dict.keys():
        data_dict[i] = np.array(data_dict[i]).reshape(dims_list[list(data_dict).index(i)]).tolist()
        data_dict[i] = torch.tensor(data_dict[i])

    # Converting to a state dict type like in the oeiginal state dict
    return collections.OrderedDict(data_dict)


def send_udp_signal(dest_port, ip_octet):
    mac_l = scapy.Ether(dst="ff:ff:ff:ff:ff:ff", src="02:42:C0:A8:00:42")
    ip_l = scapy.IP(dst="255.255.255.255", src=f"192.168.{ip_octet}.1")
    my_frame = mac_l/ip_l/scapy.UDP(dport=dest_port)
    scapy.sendp(my_frame)


def print_statistics(epoch_times, start_training_time, train_times, receive_times, load_times, write_times, losses_per_epoch):
    print(f'Average epoch train time: {get_average_of_list(epoch_times)}')
    print(f'Convergence time from worker{args["task_index"]+1}: {time.time() - start_training_time}')
    print(f'Average training time: {get_average_of_list(train_times)}')
    print(f'Average time it takes worker to receive a new model from when he is done training his model: {get_average_of_list(receive_times)}')
    print(f'Average amount of time it takes to load the new model received from the parameter server: {get_average_of_list(load_times)}')
    print(f'Average time it takes to write trained model to text file: {get_average_of_list(write_times)}')
    print(f'Final loss: {losses_per_epoch[-1]}')


# ===================================================
#                       MAIN
# ===================================================

parser = argparse.ArgumentParser(description="arguments for project",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-j", "--job_name", help="Type of job of this worker (ps or worker)")
parser.add_argument("-t", "--task_index", type=int, default=0, help="Number of worker(ps gets 0)")
parser.add_argument("-n", "--num_of_workers", type=int, default=2, help="number of workers in the distributed ML network")
parser.add_argument("-m", "--model", help="The ML model to implement")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs to run")
args = vars(parser.parse_args())

# TODO add check if correct number of args has been entered

num_of_workers = args["num_of_workers"]

# setting up
num_of_epochs = args["epochs"]
batch_size_train = 64
learning_rate = 0.01
momentum = 0.5
log_interval = 10
batch_size_test = 1000
fn_loss = nn.CrossEntropyLoss()

torch.backends.cudnn.enabled = False
torch.manual_seed(696969)

# ---------------------------- WORKER CODE ----------------------------
if args["job_name"] == 'worker':
    print(f'This is worker{args["task_index"] + 1}')

    # needed for initialization
    send_udp_signal(12345, args['task_index'] + 1)
    #

    # get data
    train_data = datasets.MNIST('./data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]))
    train_data = split_dataset(train_data, num_of_workers)[args["task_index"]]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True)

    # initialize model to train:
    network = LeNet()
    # print(f'NETWORK PARAMETERS: {list(network.named_parameters())}')
    optimizer = optim.Adam(network.parameters(), learning_rate)

    # track progress init
    train_counter = []
    test_counter = [i * len(train_loader.dataset) for i in range(num_of_epochs + 1)]
    epoch_times = []
    train_times = []            # time it takes to train on batch on model
    receive_times = []          # time it takes from when model was sent to when it was done receiving
    load_times = []             # time it takes to load new model
    write_times = []
    losses_per_epoch = []
    compute_times = []

    # ================= TRAIN =================
    # start to train model
    start_training_time = time.time()
    print("Starting training...")

    for epoch in range(1, num_of_epochs + 1):
        epoch_start_time = time.time()

        curr_state, train_counter, losses_per_epoch = train(epoch, network, optimizer, fn_loss, train_loader, train_counter, losses_per_epoch, log_interval)   # train one epoch
        print("finished training...")
        dims_list = getDimslist(curr_state)
        train_times.append(time.time() - epoch_start_time)

        # write state dict to file
        start_write_to_file_time = time.time()
        print("writing file...")
        writeFileInFormat(curr_state, f"/usr/src/app/logs/worker{args['task_index']+1}.txt")
        write_times.append(time.time() - start_write_to_file_time)
        
        # send UDP broadcast to signal switches that the file is ready to read
        compute_times.append(time.time() - epoch_start_time)
        print(f"sending signal at {time.localtime()} or {time.time()} in seconds\nCompute time is : {compute_times[-1]}")
        send_udp_signal(12345, args['task_index'] + 1)

        # wait to receive UDP broadcast
        start_wait_for_new_model_time = time.time()
        print("waiting for a signal back...")
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        recv_sock.bind(("", 12345))
        recv_sock.recvfrom(1024)
        receive_times.append(time.time() - start_wait_for_new_model_time)
        print("Recevied signal from omnet++, time to read...")

        # read new model from server text file
        start_read_file_time = time.time()
        read_state_dict = readFormatFile("/usr/src/app/logs/ps.txt", dims_list)

        # load new state dictionary
        network.load_state_dict(read_state_dict)
        load_times.append(time.time() - start_read_file_time)

        epoch_times.append(time.time() - epoch_start_time)
        print(f"Epoch {epoch} took {epoch_times[-1]}s")

    print_statistics(epoch_times, start_training_time, train_times, receive_times, load_times, write_times, losses_per_epoch)
    print(f'for {num_of_epochs} epochs and {args["task_index"]} workers:')
    print(f'Loss: {losses_per_epoch}')
    print(f'Train Times: {train_times}')
    print(f'Computation times of worker{args["task_index"]+1}: {compute_times}')
    print(f'Times it took to receive new model each epoch: {receive_times}')

    print("Program finished successfully!")
    exit(1)

# ---------------------------- PARAMETER SERVER CODE ----------------------------
elif args["job_name"] == 'ps':
    print("This is a parameter server!")

    # needed for initialization...
    send_udp_signal(49152, 0)
    #

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size_test, shuffle=True)

    computation_times = []      # times it takess the parameter server to compute new model
    read_times = []             # times it takes to read all files
    test_time = []              # time it takes to test new model
    accuracy_per_epoch = []
    test_losses = []

    epoch_num = 0
    while True:
        # wait for UDP broadcast
        print("wating for a signal...")
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        recv_sock.bind(("", 49152))
        m, _ = recv_sock.recvfrom(1024)
        recv_sock.close()

        # read received state dictionaries from files
        print("reading files...")
        files_to_read_from = glob.glob("./logs/root_switch/dict*.txt")
        print(f"files to read from:")
        if files_to_read_from:  # check if there are any files to read from
            worker_model_dicts = []
            start_read_time = time.time()
            for file in files_to_read_from:
                print(file)
                dims_list = [[6, 1, 5, 5], [6], [16, 6, 5, 5], [16], [120, 256], [120], [84, 120], [84], [10, 84], [10]] # TODO temporary fix until dynamic fix is found
                worker_model_dicts.append(readFormatFile(file, dims_list))
                os.remove(file)
            read_times.append(time.time() - start_read_time)

            # aggregate received dicts
            print("\naggregating received dicts...")
            start_compute_time = time.time()
            new_state = get_new_state(worker_model_dicts, num_of_workers)
            computation_times.append(time.time() - start_compute_time)

            # write new state dict to file
            print("writing file...")
            writeFileInFormat(new_state, "./logs/ps.txt")

            # broadcast UDP as a signal for switches to start broadcasting state
            print(f"Server is sending signal for epoch {epoch_num} at {time.localtime()} or {time.time()} in seconds")
            send_udp_signal(49152, 0)

            # test new model
            start_test_time = time.time()
            network = LeNet()
            network.load_state_dict(new_state, strict=False)
            test_accuracy = test(network, test_loader)
            accuracy_per_epoch.append(test_accuracy.tolist())
            test_time.append(time.time() - start_test_time)

            epoch_num += 1
            print(f'Finished Epoch Number {epoch_num}')
            print(f'Average time to compute new model: {get_average_of_list(computation_times)}s')
            print(f'Average time it takes the server to read all the new files: {get_average_of_list(read_times)}')
            print(f'Avergae time to test: {get_average_of_list(test_time)}s')

            if epoch_num == num_of_epochs:
                print(f'Accuracies: {accuracy_per_epoch}')
                print(f'Test losses: {test_losses}')
                break
        else:
            print("ERR: There are no files to read from... ;(")
            exit(-1)
    print('Simulation is done!')

else:
    print("ERR: Wrong job entered in argument...")
    exit(-1)
