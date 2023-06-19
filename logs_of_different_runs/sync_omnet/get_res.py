import linecache
import glob

def func(i, algos):
    for algo in algos:
        training_times = []
        epoch_times = []
        io_times = []
        ps_io_time = 0

        path = f"/home/ubuntu/Desktop/project/logs_of_different_runs/sync_omnet/{algo}{i}/worker*"
        # print(f'getting files from {path}')
        files = glob.glob(path)
        # print(files)
        for file in files:
            with open(file) as f:
                # print(f'for algorithm {algo} with {i} blue switches')
                # loop to read iterate
                # last n lines and print it
                for j, line in enumerate(f.readlines()[-13:]):
                    if j == 0:
                        epoch_times.append(float(line.split(' ')[-1]))
                    elif j == 2:
                        tt = line.split(' ')[-1]
                        training_times.append(float(tt))
                    elif j == 4:
                        iot = line.split(' ')[-1]
                        io_times.append(float(iot))
                    elif j == 5:
                        io_times[-1] += float(line.split(' ')[-1])
        print(f'For {algo}{i}')
        print(f'average epoch time: {sum(epoch_times)/len(epoch_times)} and max: {max(epoch_times)}')
        print(f'average training time: {sum(training_times)/len(training_times)} and max: {max(training_times)}')
        avg_io_t = sum(io_times)/len(io_times)
        print(f'average io times: {avg_io_t} and max: {max(io_times)}')
        with open(f"/home/ubuntu/Desktop/project/logs_of_different_runs/sync_omnet/{algo}{i}/ps.log") as f:
            for j, line in enumerate(f.readlines()[-6:]):
                if j == 1:
                    ps_io_time = float(line.split(" ")[-1])
                    print(f'parameter server io time: {ps_io_time}')
                    break
        print(f'total io is {ps_io_time + avg_io_t}')

# line = linecache.getline(r"path", 70)
algos = ['soar', 'pa', 'smc']
for i in [2, 4, 8]:
    func(i, algos)
func('', ['red','blue'])

