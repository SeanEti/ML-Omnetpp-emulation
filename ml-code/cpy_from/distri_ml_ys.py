# import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
import argparse
import socket
import time
from threading import Thread
import pickle as pkl


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
    model0 = Net()
    model0.load_state_dict(dicts[0], strict=False)

    for idx, state in enumerate(dicts):
        if idx == 0: continue
        model = Net()
        model.load_state_dict(state, strict=False)
        for param, param0 in zip(model.parameters(), model0.parameters()):
            param0.data.copy_(param.data + param0.data)

    for param in model0.parameters():
        param.data.copy_(param.data / len(dicts))

    return model0.state_dict()


def broadcast_state_dict(new_state, workers_sockets, all_workers):
    """
    Sending the new state dictionary of the model to all the workers
    """
    workers_who_shared = []
    to_send = pkl.dumps(new_state)
    for work_sock, work_addr in workers_sockets:
        workers_who_shared.append(work_addr[0])
        print(f"sending to {work_addr}...")
        if work_sock.sendall(to_send) == -1:
            print(f"ERR: Failed to send new state of model to {work_addr} while broadcasting...")

    # check if there is a worker who didnt send his model -> need to establish new tcp connection with him
    workers_who_skipped = [item for item in all_workers if item[0] not in workers_who_shared]
    if len(workers_who_skipped) > 0:
        # create connections and send new model
        for worker_addr in workers_who_skipped:
            new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if new_sock.connect_ex(worker_addr) == 0:
                print(f"sending to {worker_addr}...")
                if work_sock.sendall(to_send) == -1:
                    print(f"ERR: Failed to send new state of model to {worker_addr} while broadcasting...")
            else:
                print(f"couldn't send new model to {worker_addr}")
            new_sock.close()


def handle_new_worker_connection(work_sock, addr, worker_model_dict):
    """
    Receives model from worker in addr
    """
    print(f"Receives model from {addr}...")
    # receive model of worker
    data = recvall(work_sock)
    message = pkl.loads(data)                       # deserialize model state dict
    worker_model_dict[(work_sock, addr)] = message


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
for idx, worker in enumerate(workers):
    work_ip, work_port = worker.split(':')
    work_port = int(work_port)
    workers[idx] = (work_ip, work_port)
num_of_workers = len(workers)

parameter_server = args["server"]
serv_ip, serv_port = parameter_server.split(':')    # get IP and port of parameter-server
serv_port = int(serv_port)

print(f'Received IPs:\nWorkers: {workers}\nParameter Server: {parameter_server}')

# setting up
num_of_epochs = 10
batch_size_train = 64
learning_rate = 0.01
momentum = 0.5
log_interval = 10
batch_size_test = 1000

torch.backends.cudnn.enabled = False
torch.manual_seed(696969)

# ---------------------------- WORKER CODE ----------------------------
if args["job_name"] == 'worker':
    print(f'This is worker{args["task_index"] + 1}')

    # get data
    train_data = datasets.MNIST('./data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]))
    train_data = split_dataset(train_data, num_of_workers)[args["task_index"]]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True)

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
    epoch_times = []

    # start to train model
    start_training_time = time.time()
    print("Starting training...")
    for epoch in range(1, num_of_epochs + 1):
        epoch_start_time = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        curr_state = train(epoch, network, optimizer, train_loader, log_interval)   # train one epoch

        # serialize data to send
        state_string = pkl.dumps(curr_state, -1)
        
        # send gradients to server
        # TODO need to add timeout maybe for smart switch to work...
        if sock.connect_ex((serv_ip, serv_port)) == 0:
            print("sending model to server...")
            sock.sendall(state_string)
            print("sent model to server! now waiting to hear back...")
            data = recvall(sock)
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        else:   # listen for server connection
            print(f'server won\'t connect me so im binding {workers[args["task_index"]]}')
            sock.bind(workers[args["task_index"]])
            print("Couldn't connect to server so waiting for connection from server...")
            sock.listen(1)
            # TODO maybe need to reset timeout here from first TODO
            conn, _ = sock.accept()
            data = recvall(conn)
            conn.shutdown(socket.SHUT_RDWR)
            conn.close()

        # wait to receive new params from server
        print("Received updated model from server...")
        new_state_dict = pkl.loads(data)
        network.load_state_dict(new_state_dict)     # update network

        epoch_times.append(time.time() - epoch_start_time)
        print(f"Epoch {epoch} took {epoch_times[-1]}s")
        # test new network  
        test(network, test_loader)

    print(f'Average epoch train time: {sum(epoch_times) / len(epoch_times)}\nConvergence time from worker{args["task_index"]+1}: {time.time() - start_training_time}')
    print("Program finished successfully!")
    exit(1)

# ---------------------------- PARAMETER SERVER CODE ----------------------------
elif args["job_name"] == 'ps':
    print("This is a parameter server!")
    # bind socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((serv_ip, serv_port))

    # num_of_workers_who_shared_model = 0
    worker_model_dict = dict()

    # listen for incoming datagrams
    sock.listen(num_of_workers)
    threads = []
    while True:
        print("Listening... waiting for workers to train...")
        worker_model_dict = dict()
        sock.settimeout(None)
        conn, addr = sock.accept()
        worker_t = Thread(target=handle_new_worker_connection, args=(conn, addr, worker_model_dict))
        threads.append(worker_t)
        worker_t.start()
        print("Accepted first worker... now waiting for rest")
        for i in range(num_of_workers - 1):
            sock.settimeout(6)
            try:
                conn, addr = sock.accept()
            except socket.timeout:
                print("timeout...")
                break
            print(f"accepting connection from: {addr}")
            worker_t = Thread(target=handle_new_worker_connection, args=(conn, addr, worker_model_dict))
            threads.append(worker_t)
            worker_t.start()
        
        print("waiting for threads to finish reading models")
        for t in threads:
            t.join()

        print("getting new aste dictionary from received dics")
        new_state = get_new_state(worker_model_dict)
        broadcast_state_dict(new_state, list(worker_model_dict.keys()), workers)
else:
    print("ERR: Wrong job entered in argument...")
    exit(-1)
