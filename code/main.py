import argparse
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import yaml
import os
from tqdm import tqdm

import device
import helper

parser = argparse.ArgumentParser()
# File name
parser.add_argument("--file_root", type=str, default="./output_basic/")
parser.add_argument("--file_name_device", type=str, default="./output_basic/Device_spec", help="name of csv file")
parser.add_argument("--file_name_round", type=str, default="./output_basic/FL_output", help="name of csv file")
parser.add_argument("--file_name_graph", type=str, default="./output_basic/FL_accuracy_Graph", help="name of graph png file")

# Parameter
parser.add_argument("--edge_num", type=int, default="3", help="number of edge devices")
parser.add_argument("--round_num", type=int, default="20", help="number of rounds")
parser.add_argument("--model", type=str, default="MobileNet_V2", help="global model name")
parser.add_argument("--dataset", nargs='+', type=str, default=["MNIST","CIFAR10","CelebA"], help="dataset name(MNIST,CIFAR10,CelebA")
parser.add_argument("--server_comm", type=str, default="1Gbps", help="server communication method")
parser.add_argument("--client_comm", nargs='+', type=str, default=["LTE", "5G"], help="edge device communication method")
parser.add_argument("--l_agent_comm", nargs='+', type=str, default=["LTE", "5G", "WIFI", "1Gbps"], help="learning agent communication method")
parser.add_argument("--l_agent_num", type=int, default=3, help="number of maximum learning agent")

parser.add_argument("--a", type=float, default=0, help="minimum accuracy")
parser.add_argument("--b", type=float, default=0.8, help="maximum accuracy")
parser.add_argument("--p", type=float, default=0.02, help="weight of local training epoch")
parser.add_argument("--q", type=float, default=0.0001, help="weight of participated client")

parser.add_argument("--accuracy", nargs='+', type=float, default=None)
parser.add_argument("--predict_round", type=int, default=None)

opt = parser.parse_args()
print(opt)

os.makedirs(opt.file_root, exist_ok = True)

def custom_predict(target):
    result_list = []                    # ????????? ??????
    bw_list = []                        # ???????????? ???????????? b,w ??????
    a = target[0]                       # ????????? ?????????
    b_range = np.arange(0,1,0.001)      # ?????? ????????? ??????
    w_range = np.arange(0.001,3,0.001)  # ????????? ????????? ??????

    for b in tqdm(b_range, desc='Start predict accuracy...'):
        for w in w_range:
            predict = []
            for x in range(len(target)):
                y = helper.custom_sigmoid(x*w,a,b)
                predict.append(y)
            error = [abs(target-predict) for target, predict in zip(target, predict)]
            error_sum = sum(error)
            result_list.append(error_sum)
            bw_list.append([b,w])

    min_error = min(result_list)
    min_error_idx = result_list.index(min_error)
    min_error_bw = bw_list[min_error_idx]

    p_predict = []
    b = min_error_bw[0]
    num = opt.predict_round
    for n in range(num):
        y = helper.custom_sigmoid(n*min_error_bw[1],a,b)
        p_predict.append(y)
    return p_predict

def show_custom_predict(p_predict,target, predict_round):
    X = np.arange(0,predict_round,1)
    X_t = np.arange(0, len(target), 1)
    plt.plot(X, p_predict, label='Predict')
    plt.plot(X_t, target, color='green',marker='o', label='Target')
    plt.title('Predict Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Predict_accuracy.png')
    plt.show()

if opt.accuracy is not None:
    p_predict = custom_predict(opt.accuracy)
    show_custom_predict(p_predict, opt.accuracy, opt.predict_round)
    exit()

# ?????? ?????? ??????
with open('yaml/device.yaml') as device_yaml:
    device_spec = yaml.load(device_yaml, Loader=yaml.FullLoader)
    gpu = device_spec['GPU']
    soc = device_spec['SoC']

# ?????? ?????? ??????
with open('yaml/communication.yaml') as comm_yaml:
    comm_list = yaml.load(comm_yaml, Loader=yaml.FullLoader)
    server_comm = comm_list['communication'][opt.server_comm]
    cl_list = []
    la_list = []
    for cl in opt.client_comm:
        cl_list.append((cl,comm_list['communication'][cl]))
    for ag in opt.l_agent_comm:
        la_list.append((ag,comm_list['communication'][ag]))

# ?????? ?????? ??????
model_name = opt.model
with open('yaml/model.yaml') as model_yaml:
    model_list = yaml.load(model_yaml, Loader=yaml.FullLoader)
    model_spec = model_list['model'][model_name] # {'size': 14, 'unit': 'MB'}
    model_size = model_spec['size'] # 14

# dataset ?????? ??????
with open('yaml/dataset.yaml') as dataset_yaml:
    dataset_list = yaml.load(dataset_yaml, Loader=yaml.FullLoader)
    mnist = dataset_list['dataset']['MNIST']
    cifar10 = dataset_list['dataset']['CIFAR10']
    celeba = dataset_list['dataset']['CelebA']


# ??????????????? ??? ????????????????????? ????????? ?????? ??????
clients = list(soc.keys())
agents = list(gpu.keys()) + list(soc.keys())
total_device = list(soc.keys()) + list(gpu.keys())

### Device Spec
print(f'[Step1] Device Information')
## 1. Server
# ?????? ?????? : Mobilenet, Inception, U-Net, SRGAN
# ?????? ?????? : LTE, 5G, Ethernet 5M, Ethernet 1G
print(f'  1. Server Device Information')
print(f'     Global Model: {model_name}')
print(f'     Model Spec: {model_spec}')
print(f'     Communication: {opt.server_comm},{server_comm}')
global_model = device.server(model_size,server_comm['throughput'])

# server dataframe
df_global = global_model.df_global(global_model.item())

## 2. Device (Client)
# ?????? : server?????? ?????? ??????
# ?????? ?????? : LTE, 5G, Ethernet 5M, Ethernet 1G
# ?????? ???????????? : MNIST, CIFAR10, CelebA
print(f'  2. Edge Device({opt.edge_num} Client) Information')
client_spec = []
client_name = []

for i in range(opt.edge_num):
    print('')
    print(f'     * Client{i+1} Information')

    # ??????????????? ?????? ?????? : ????????? ???????????? ????????? ??? ????????? ????????? ???????????? ??????
    if i+1 > len(cl_list) :
        client_comm_spec = random.sample(cl_list,1)
        client_comm = client_comm_spec[0][1]['throughput']
        print(f'       Communication: {client_comm_spec}')
    else :
        client_comm_spec = cl_list[i]
        client_comm = client_comm_spec[1]['throughput']
        print(f'       Communication: {client_comm_spec}')

    # ????????????????????? ????????? ???????????? ??????
    data = random.sample(opt.dataset, 1)[0]
    dataset_spec = dataset_list['dataset'][data]
    dataset_size = dataset_spec['volume']
    print(f'       Dataset: {data}, {dataset_spec}')

    dev = random.sample(clients, 1)[0]
    if dev in list(gpu.keys()):
        category = gpu[dev]
    elif dev in list(soc.keys()):
        category = soc[dev]

    compute_rate = category[model_name]
    print(f'       Coumputing Power: {dev}, {compute_rate} ms')

    client = device.edge(model_size, client_comm, dataset_size, compute_rate)
    client_name.append('Client' + str(i + 1))
    client_spec.append(client.item())
df_edge = client.df_edge(client_spec, client_name)

## 3. Learning agent
print(f'  3. Learning Agent Information')
agent_spec = []
agent_name = []

l_agent_num_count = []
for index, name in enumerate(client_name):

    dataset = df_edge.at[name, 'Dataset_Size']
    l_agent_num = random.randrange(0, opt.l_agent_num)
    l_agent_num_count.append(l_agent_num)
    print(f'     * {name} {l_agent_num} Learning Agent Information')
    for agent_num in range(l_agent_num):
        print(f'       ** Learning Agent {agent_num+1} Info')
        # LearningAgent ?????? ?????? : ????????? ?????? ??? ???????????? ??????
        l_comm_spec = random.sample(la_list,1)
        print(f'          Communication: {l_comm_spec}')
        l_comm = l_comm_spec[0][1]['throughput']

        l_agent = random.sample(agents,1)[0]
        if l_agent in list(gpu.keys()):
            category = gpu[l_agent]
        elif l_agent in list(soc.keys()):
            category = soc[l_agent]
        compute_rate = category[model_name]
        print(f'          Computing Power: {l_agent}, {compute_rate} ms')
        agent = device.learning_agent(model_size, l_comm, dataset, compute_rate)
        agent_name.append(name + '_agent' + str(agent_num + 1))
        agent_spec.append(agent.item())
if any(l_agent_num_count) == False:
    df = pd.concat([df_global, df_edge], axis=0)
    df.to_csv(opt.file_name_device + '.csv')
    print('agent false')
else:
    df_agent = agent.df_agent(agent_spec, agent_name)
    df = pd.concat([df_global, df_edge, df_agent], axis=0)
    df.to_csv(opt.file_name_device + '.csv')
    print('agent true')

### Round
print('')
print('')
print(f'[Step2] Simulation')
total_time_item = [] # ?????????????????? ????????? ??? ??????
round_time_item = []
aggre_acc_list = []
x_total = 0 # x of sigmoid

df_round = pd.DataFrame()
for r in range(1, opt.round_num+1):
    ### reset information each round
    output_index = []    #
    output_column = []   #
    round_item = []      #
    round_time_list = [] # ????????? ??????
    round_dataset = []
    round_iteration = []

    ### ?????? round
    round_name = 'Round' + str(r)
    output_index.append(round_name)
    print(f'  Round {r}')

    ## 1. Edge device Select : round ?????? device ??????
    # device ??????
    print(f'    1. Edge device Select - Total device: {client_name}')
    random_num = helper.random_integer(1, len(client_name))
    device_name_sample = random.sample(client_name, random_num)
    device_name_sample.sort()
    print(f'       Selected device: {device_name_sample}')

    # device ??????(?????????,??????) ??????
    target_device = []  # target_device : (index,device name)
    for g, device_g in enumerate(client_name):
        for device_s in device_name_sample:
            if device_s == device_g:
                target_device.append([g + 1, device_s])

    ## 2. deploy : global model??? edge device??? ?????? (?????? ??????(s) = Global model size(MB) / Communication(MB/s)
    print(f'    2. Model deploy')
    deploy_time = global_model.time
    print(f'       Deploy time: {deploy_time} s')
    output_column.append('model deploy time')
    round_item.append(deploy_time)

    ## 3. Select device local train : edge device?????? ??????(?????? ?????? = time + local train time)
    # Leaning Agent??? ?????? ?????? ?????? ?????? ????????? ???????????? ????????? ????????? ?????????
    print(f'    3. Local train & Send to server ({len(device_name_sample)} device)')
    print(f'        -> Learning time(ms) = learning time(iter 1) * iteration')
    print(f'        -> Device(Edge/Learning Agent) to Server = Learning time(s) + send time(s)')
    print(f'        ---> Edge device to Server : learning time(s) + (edge device) send time(s)')
    print(f'        ---> Learning agent to Server : (edge device) send time(s) + learning time(s) + (learning agent) send time(s)')

    for sample in target_device:
        target_device_name = sample[1]
        max_iter_value = 4  # iteration ?????? (50??? ??????. e.g 50, 100, 150, 200)
        iteration = helper.random_integer(1, max_iter_value) * 50  # iteration

        output_column.append(target_device_name + '_iter')
        round_item.append(iteration)

        df_index = list(df.index)  # global, client, learning agent
        find = []
        la_time = 0
        la = 'None'
        for z in df_index:
            if sample[1] + '_' in z:  # learning agent??? ?????? ??????
                find.append(z)
                use_status = df.loc[z].Use_Status

                output_column.append(z + 'use_status')
                round_item.append(use_status)

                # learning agent??? ?????? ????????? ?????? : edge ?????? + ?????? + server ??????
                # la_time : learning agent??? ????????? ????????? ??????
                # la : ????????? learning agent
                if use_status == 1:
                    if la_time == 0:  # ??????????????? learning agent??? ????????? ??????
                        la_time = df.loc[target_device_name].Transmission_Delay + df.loc[z].Learning_Time * iteration / 1000 + df.loc[z].Transmission_Delay
                        la = z
                    else:  # ????????? ??????????????? learning agent??? ?????? ??????
                        temp_la_time = df.loc[target_device_name].Transmission_Delay + df.loc[z].Learning_Time * iteration / 1000 + df.loc[z].Transmission_Delay
                        if la_time > temp_la_time:
                            la_time = temp_la_time
                            la = z
        round_dataset.append(df.loc[target_device_name].Dataset_Size)
        round_iteration.append(iteration)

        # learning agent??? ?????? ?????? : ?????? + server ??????
        edge_time = df.loc[target_device_name].Learning_Time * iteration / 1000 + df.loc[target_device_name].Transmission_Delay
        print(f'        * {sample[1]} iteration: {iteration}')
        print(f'            Edge device to Server : {df.loc[target_device_name].Learning_Time * iteration / 1000:.3f} + {df.loc[target_device_name].Transmission_Delay:.3f} = {edge_time:.3f} s')
        print(f'            Learning Agent to Server : {la}, la_time:{la_time:.3f} s')

        # edge device ???????????? : ???????????? + ????????? ????????????
        # ?????????????????? ???????????? : ????????????????????? ???????????? + ???????????? + ????????? ????????????
        if edge_time <= la_time or la_time == 0:
            use_device = target_device_name
            send_time = edge_time
        else:
            use_device = la
            send_time = la_time
        round_time_list.append(send_time)
        print(f'          Use_device:{use_device} / Send to Server:{send_time:.3f} s')

        output_column.append(target_device_name + '_send_time')
        output_column.append(target_device_name + 'Learning Agent')
        output_column.append(target_device_name + 'Learning Agent send_time')
        output_column.append(target_device_name + 'Using Device')
        output_column.append(target_device_name + 'Using Device send_time')
        round_item.append(edge_time)
        round_item.append(la)
        round_item.append(la_time)
        round_item.append(use_device)
        round_item.append(send_time)

    # round ?????? ?????? ?????? : ????????? ????????? ???????????? ??? ?????? ?????? ????????? ????????? ????????? ???????????? ??????
    round_time = max(round_time_list)
    round_time_item.append(round_time)

    # total ?????? ??????
    total_time = deploy_time + round_time
    total_time_item.append(total_time)
    print('')
    print(f'        Total Round time = {total_time:.3f} s')

    # accuracy
    round_dataset = np.array(round_dataset)
    round_iteration = np.array(round_iteration) * opt.p

    min_acc = opt.a
    max_acc = opt.b
    x = round_dataset * round_iteration
    x_weight_sum = np.sum(x) * ((len(device_name_sample)*opt.q)/len(client_name))
    x_total = x_total + x_weight_sum
    aggre_acc = helper.custom_sigmoid(x_total, min_acc, max_acc)
    aggre_acc_list.append(aggre_acc)
    print(f'        Aggregation accuracy: {aggre_acc:.3f}')

    # DataFrame ??????
    round_item = np.reshape(round_item, (1, len(round_item)))
    df_temp = pd.DataFrame(round_item, columns=output_column, index=output_index)
    print('')
    df_round = pd.concat([df_round, df_temp], join='outer', axis=0)

df_round['round time'] = round_time_item
df_round['total round time'] = total_time_item
df_round['aggregation accuracy'] = aggre_acc_list
print(f"???????????? ??? ?????? ?????? : {df_round['total round time'].sum():.3f} s")
print(f"???????????? ?????? Accuracy : {df_round['aggregation accuracy'][-1]:.3f}")
df_round_copy = df_round.transpose()
df_round.to_csv(opt.file_name_round + '.csv')
df_round_copy.to_csv(opt.file_name_round + '_trans.csv')

# Accuracy Graph
x_time = 0
X_list = []
for temp in total_time_item:
    x_time  = x_time + temp/1000
    X_time = round(x_time,2)
    X_list.append(X_time)

X = np.arange(1,opt.round_num+1,1)
y = aggre_acc_list


plt.subplot(2, 1, 1)                # nrows=2, ncols=1, index=1
plt.plot(X, y, 'o-')
plt.title('X=round')
plt.ylabel('Accuracy')

plt.subplot(2, 1, 2)                # nrows=2, ncols=1, index=2
plt.plot(X_list, y, '.-')
plt.title('X=time')
plt.xlabel('time (s)')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig(opt.file_name_graph + '.png')
plt.show()