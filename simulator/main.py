import argparse
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import yaml

import device
import helper

parser = argparse.ArgumentParser()
# File name
parser.add_argument("--file_name_device", type=str, default="Device_spec", help="name of csv file")
parser.add_argument("--file_name_round", type=str, default="FL_output", help="name of csv file")
parser.add_argument("--file_name_graph", type=str, default="FL_accuracy_Graph", help="name of graph png file")
# Parameter
parser.add_argument("--edge_num", type=int, default="3", help="number of edge devices")
parser.add_argument("--round_num", type=int, default="3", help="number of rounds")
# parser.add_argument("--model", type=str, default=None, help="global model name")
parser.add_argument("--model", type=str, default="MobileNet_V2", help="global model name") # fix
parser.add_argument("--model_size", type=int, default=None, help="custom model size")

# parser.add_argument("--dataset", nargs='+', type=str, default=None, help="dataset name(MNIST,CIFAR10,CelebA")
parser.add_argument("--dataset", nargs='+', type=str, default=["MNIST","CIFAR10","CelebA"], help="dataset name(MNIST,CIFAR10,CelebA") # fix
parser.add_argument("--dataset_vol", type=int, default=None, help="custom dataset volume")
parser.add_argument("--dataset_imgs", type=int, default=None, help="custom dataset imgs")
parser.add_argument("--dataset_size", nargs='+', type=int, default=None, help="custom dataset size")

# parser.add_argument("--server_comm", type=str, default=None, help="server communication method")
parser.add_argument("--server_comm", type=str, default="LTE", help="server communication method") # fix
# parser.add_argument("--client_comm", nargs='+', type=str, default=None, help="edge device communication method")
parser.add_argument("--client_comm", nargs='+', type=str, default=["LTE", "5G"], help="edge device communication method") # fix
# parser.add_argument("--l_agent_comm", type=str, default=None, help="learning agent communication method")
parser.add_argument("--l_agent_comm", nargs='+', type=str, default=["LTE", "5G", "Wifi", "lan_1G"], help="learning agent communication method")
# parser.add_argument("--comm_speed", type=int, default=None, help="custom communication speed")
parser.add_argument("--comm_speed", type=int, default=10, help="custom communication speed") # fix

parser.add_argument("--l_agent_num", type=int, default=2, help="number of learning agent")
opt = parser.parse_args()
print(opt)

# device 목록 로드
with open('device.yaml') as f:
    device_spec = yaml.load(f, Loader=yaml.FullLoader)
    gpu = device_spec['GPU']
    soc = device_spec['SoC']
    phone = device_spec['Phone']
    iot = device_spec['IoT']
# print(device_spec)
device_list = list(gpu.keys()) + list(soc.keys()) + list(phone.keys()) + list(iot.keys())
agent_list = list(gpu.keys()) + list(soc.keys()) + list(iot.keys())
# print(device_list) # 사용 가능한 기기 이름
# print(device_spec['GPU']) # GPU에 해당하는 모든 내용
# print(device_spec['GPU'].keys()) # GPU 모델명만 출력
# print(device_spec['GPU']['NVIDIA TITAN RTX'].keys()) # NVIDIA TITAN RTX 학습 모델명만 출력

### Device Spec
## 1. Server
# 모델 설정 : Mobilenet, Inception, Vgg, SRGAN
# 통신 설정 : LTE, 5G, 유선5M, 유선1G
model_name = opt.model
model = helper.model(opt.model, opt.model_size)
server_comm = helper.communication(opt.server_comm, opt.comm_speed)
global_model = device.server(model, server_comm)
# server dataframe
df_global = global_model.df_global(global_model.item())

## 2. Device (Client)
# 모델 : server에서 보낸 모델
# 통신 설정 : LTE, 5G, 유선5M, 유선1G
# 로컬 데이터섯 : MNIST, CIFAR10, CelebA
#client_comm = opt.client_comm
client_spec = []
client_name = []
for i in range(opt.edge_num):
    # 클라이언트 통신 설정 : 입력한 순서대로 기입한 후 입력이 없으면 고정 값으로 지정
    if i+1 > len(opt.client_comm) :
        client_comm = helper.communication('mycomm', opt.comm_speed)
    else :
        client_comm = helper.communication(opt.client_comm[i], opt.comm_speed)

    # local dataset 설정 : 입력한 리스트에서 랜덤으로 불러옴
    data = random.sample(opt.dataset, 1)[0]
    dataset = helper.dataset(str(data), opt.dataset_vol, opt.dataset_imgs, opt.dataset_size)

    # print(f'통신속도:{client_comm}')
    # print(f'데이터셋:{data}')
    # print(f'데이터셋 설명:{dataset}')

    dev = random.sample(device_list, 1)[0]
    if dev in list(gpu.keys()):
        category = gpu[dev]
        # compute_rate = gpu[dev][model_name]
    elif dev in list(soc.keys()):
        category = soc[dev]
        # compute_rate = soc[dev][model_name]
    elif dev in list(phone.keys()):
        category = phone[dev]
        # compute_rate = phone[dev][model_name]
    elif dev in list(iot.keys()):
        category = iot[dev]
        # compute_rate = iot[dev][model_name]
    compute_rate = category[model_name]

    # print(f'장치:{dev}')
    # print(f'이미지 처리 성능:{compute_rate}')

    client = device.edge(model, client_comm, dataset[0], compute_rate)
    client_name.append('Client'+str(i+1))
    client_spec.append(client.item())
df_edge = client.df_edge(client_spec,client_name)

## 3. Learning agent
agent_spec = []
agent_name = []

for index, name in enumerate(client_name):
    dataset = df_edge.at[name, 'data']
    for agent_num in range(opt.l_agent_num):
        if agent_num + 1 > len(opt.l_agent_comm):
            l_comm = helper.communication('mycomm', opt.comm_speed)
        else:
            ran_comm = random.sample(opt.l_agent_comm,1)[0]
            l_comm = helper.communication(ran_comm, opt.comm_speed)

        l_aent = random.sample(agent_list,1)[0]
        if l_aent in list(gpu.keys()):
            category = gpu[l_aent]
        elif l_aent in list(soc.keys()):
            category = soc[l_aent]
        elif l_aent in list(phone.keys()):
            category = phone[l_aent]
        elif l_aent in list(iot.keys()):
            category = iot[l_aent]
        compute_rate = category[model_name]
        agent = device.learning_agent(model, l_comm, dataset, compute_rate)
        agent_name.append(name + '_agent' + str(agent_num + 1))
        agent_spec.append(agent.item())
df_agent = agent.df_agent(agent_spec, agent_name)

df = pd.concat([df_global, df_edge, df_agent], axis=0)
df.to_csv('../' + opt.file_name_device + '.csv')
print(df)
### Round
last_accuracy_list = []
old_accuracy_list = []
last_aggre_acc = 0
total_time_item = [] # 연합학습에서 소요된 총 시간
round_time_item = []
aggre_acc_list = []

df_round = pd.DataFrame()
for round in range(1, opt.round_num+1):
    ### reset information each round
    output_index = []    #
    output_column = []   #
    round_item = []      #
    accuracy_list = []   # 정확도
    round_time_list = [] # 라운드 시간

    ### 현재 round
    round_name = 'Round' + str(round)
    output_index.append(round_name)
    print('')
    print(f'* Round {round}')

    ## 1. Edge device Select : round 참여 device 선택
    # device 선택
    print(f'  1. Edge device Select')
    print(f'     Total device: {client_name}')
    random_num = helper.random_integer(1, len(client_name))
    device_name_sample = random.sample(client_name, random_num)
    device_name_sample.sort()

    # device 정보(인덱스,이름) 추출
    target_device = []  # target_device : (index,device name)
    for g, device_g in enumerate(client_name):
        for device_s in device_name_sample:
            if device_s == device_g:
                target_device.append([g + 1, device_s])
    print(f'     선택된 디바이스: {device_name_sample}')
    # print(f'     선택된 디바이스: {target_device}')

    ## 2. deploy : global model을 edge device에 전송 (소요 시간 = deploy_time)
    print(f'  2. Model deploy')
    deploy_time = global_model.time
    print(f'     Deploy time: {deploy_time}')
    output_column.append('model deploy time')
    round_item.append(deploy_time)

    ## 3. Select device local train : edge device에서 학습(소요 시간 = time + local train time)
    # Leaning Agent가 있을 경우 사용 가능 여부를 확인하여 학습을 옮겨서 실행행
    print(f'  3. Local train & Send to server ({len(device_name_sample)} device)')
    print(f'     Required time = learning time * iteration//10 + send time')  # 왜 //10 하는지 확인

    # for index_s, sample in enumerate(target_device): # 랜덤하게 선택한 device의 인덱스와 이름
    for sample in target_device:
        # print(sample) # sample[0] : 인덱스 / sample[1] : sampling device name
        target_device_name = sample[1]
        max_iter_value = 4  # 후에 조정 - iteration 조정
        iteration = helper.random_integer(1, max_iter_value) * 50  # iteration
        print(f'       {sample[1]} iteration: {iteration}')

        output_column.append(target_device_name + '_iter')
        round_item.append(iteration)

        # learning agent 존재
        df_index = list(df.index)
        find = []
        la_time = 0
        la = 'None'
        for z in df_index:
            if sample[1] + '_' in z:  # learning agent가 있을 경우
                find.append(z)
                print(find)
                use_status = df.loc[z].use_status

                output_column.append(z + 'use_status')
                round_item.append(use_status)

                if use_status == 1:  # learning agent가 사용 가능한 경우
                    if la_time == 0:  # 사용가능한 learning agent가 없었던 경우
                        la_time = df.loc[z].time + df.loc[z].learning_time
                        la = z
                    else:  # 이전에 사용가능한 learning agent가 있던 경우
                        temp_la_time = df.loc[z].time + df.loc[z].learning_time
                        if la_time > temp_la_time:
                            la_time = temp_la_time
                            la = z
        # print('index:', df_index) # 전체 device
        # print('find: ', find)     # 사용 가능한 learning agent
        edge_time = df.loc[target_device_name].time + df.loc[target_device_name].learning_time
        print('       * edge_time : (edge device) send time + learning time')
        print('       * la_time : (learning agent) send time + learning time')
        print(f'       edge_time:{edge_time}')
        print(f'       learning agent:{la}, la_time:{la_time}')
        if edge_time <= la_time or la_time == 0:
            use_device = target_device_name
            send_time = edge_time
        else:
            use_device = la
            send_time = la_time
        round_time_list.append(send_time)
        # send time = send time + learning time
        print(f'       use_device:{use_device} / send_time:{send_time}', )

        # device accuracy
        # device name : use_device
        # accuracy = (data size * iteration) / f(=적절한 값)
        # f = data size + iteration
        f = df.loc[use_device].data + iteration
        # round1 : 이전 round의 accuracy가 존재하지 않음
        if round == 1:
            accuracy = df.loc[use_device].data * iteration // f
            accuracy_list.append(accuracy)
            # print(df.loc[use_device].data_size)
            # print(iteration)
            # print(f)
            # print('acc: ', accuracy)
        # round2 이상 : 이전 round의 accuracy를 고려하여 accuracy 계산
        else:
            last_aggre_acc = aggre_acc_list[-1]
            accuracy = (df.loc[use_device].data * iteration // f) + last_aggre_acc
            accuracy_list.append(accuracy)
        #     print('last aggre : ', last_aggre_acc)
        # print(df.loc[use_device].data_size)
        # print(iteration)
        # print(f)
        # print('acc: ', accuracy)

        output_column.append(target_device_name + '_send_time')
        output_column.append(target_device_name + 'Learning Agent')
        output_column.append(target_device_name + 'Learning Agent send_time')
        output_column.append(target_device_name + 'Using Device')
        output_column.append(target_device_name + 'Using Device send_time')
        output_column.append(target_device_name + 'Accuracy')
        round_item.append(edge_time)
        round_item.append(la)
        round_item.append(la_time)
        round_item.append(use_device)
        round_item.append(send_time)
        round_item.append(accuracy)

    # round 소요 시간 계산 : 학습에 참여한 디바이스 중 가장 늦게 학습과 전송을 완료한 디바이스 기준
    # print(f'round_time_list : {round_time_list}')
    round_time = max(round_time_list)
    print(f'     Required time = {round_time}')

    round_time_item.append(round_time)
    # output_column.append('round time')
    # round_item.append(round_time)

    # total 소요 시간
    total_time = deploy_time + round_time
    total_time_item.append(total_time)
    # print(np.shape(round_item))
    # print(np.shape(output_column))
    # print(np.shape(output_index))
    # print(round_item)
    # print(output_column)
    # print(output_index)

    # accuracy aggregation = sum(device accuracy) // aggregation f(=적당한 값)
    aggre_f = 3 * len(client_name) - len(device_name_sample)  # device 개수가 많을수록 높은 accuracy가 계산되도록
    if round == 1:
        aggre_acc = sum(accuracy_list) // aggre_f
    else:
        aggre_acc = sum(accuracy_list) // aggre_f + aggre_acc_list[-1]
    aggre_acc_list.append(aggre_acc)
    print(f'     aggre_acc: {aggre_acc}')
    # print('aggre_acc_list: ', aggre_acc_list)

    # DataFrame 생성
    round_item = np.reshape(round_item, (1, len(round_item)))
    df_temp = pd.DataFrame(round_item, columns=output_column, index=output_index)
    # df_temp = df_temp.reset_index()
    print('')
    print(df_temp)
    df_round = pd.concat([df_round, df_temp], join='outer', axis=0)

df_round['round time'] = round_time_item
df_round['total round time'] = total_time_item
df_round['aggregation accuracy'] = aggre_acc_list
print(df_round)
print(f"연합학습 총 소요 시간 : {df_round['total round time'].sum()}")
print(f"연합학습 최종 Accuracy : {df_round['aggregation accuracy'][-1]}")
df_round_copy = df_round.transpose()
print(df_round_copy)
df_round.to_csv('../' + opt.file_name_round + '.csv')
df_round_copy.to_csv('../' + opt.file_name_round + '_trans.csv')

# Accuracy Graph
X = list(df_round.index)
y = list(df_round['aggregation accuracy'])
plt.plot(X, y, linestyle='dashed', marker='.', color='violet')
plt.xlabel('Round')
plt.ylabel('Accuracy')
# plt.show()
plt.savefig('../' + opt.file_name_graph + '.png')


# if __name__ == "__main__":
#     print('1. Global model')
#     print(df_global)
#     print(df_edge)
#     print(df_agent)
#     print(df)

