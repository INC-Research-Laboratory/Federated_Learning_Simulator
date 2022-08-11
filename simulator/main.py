import argparse
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import device
import helper

parser = argparse.ArgumentParser()
parser.add_argument("--file_name_device", type=str, default="Device_spec", help="name of csv file")
parser.add_argument("--file_name_round", type=str, default="FL_output", help="name of csv file")
parser.add_argument("--file_name_graph", type=str, default="FL_accuracy_Graph", help="name of graph png file")
parser.add_argument("--edge_num", type=int, default=2, help="number of edge device")
parser.add_argument("--round_num", type=int, default=3, help="number of round")
parser.add_argument("--model_size", type= int, default=20 , help="size of model")
parser.add_argument("--server_speed", type=int , default=10 , help="speed of server")
parser.add_argument("--max_edge_speed", type=int , default=10 , help="maximum speed of edge device")
parser.add_argument("--edge_data", type=int , default=3 , help="size of data")
parser.add_argument("--max_learning_agent_speed", type=int , default=20 , help="maximum speed of learning agent")
parser.add_argument("--max_learning_agent_num", type=int , default=2 , help="maximum number of learning agent")
opt = parser.parse_args()
print(opt)

# Device Spec
# 1. Global (Server)
# model_size = 20  # 후에 변경
# server_speed = 2 # 후에 변경
global_model = device.server(opt.model_size, opt.server_speed)
# 정보 확인
# global_model.info()
global_item_list = global_model.item()
global_shape = np.shape(global_item_list)
global_item_list = np.reshape(global_model.item(),(global_shape[0],1))
global_item_list = np.transpose(global_item_list)

global_index = ['model_size', 'send_speed', 'send_time']
df_global = pd.DataFrame(global_item_list, columns=global_index, index=['global'])
# df_global = pd.DataFrame(global_item_list, columns=['global'])
print('1. Global model')
print(df_global)

# 2. Device (Client)
# edge_num = 2
# max_edge_speed = 10
# edge_data = 10
device_list = []
device_name = []
for i in range(opt.edge_num):
    edge_device = device.edge(opt.model_size, opt.max_edge_speed, opt.edge_data)
    device_name.append('edge'+str(i+1))
    device_list.append(edge_device.item())
    # 정보 확인
    # edge_device.info()
#device_list = np.transpose(device_list)
#print(device_list)
edge_index = ['model_size', 'send_speed', 'send_time', 'data_size', 'compute_rate', 'learning_time']
df_edge = pd.DataFrame(device_list, columns=edge_index, index=device_name) #, columns=['edge1']
# df_edge = pd.DataFrame(device_list, columns=device_name) #, columns=['edge1']
print('')
print('2. Edge device')
print(df_edge)

# 3. Learning Agent
# learning_agent_num = 1
# max_learning_agent_speed = 20
agent_index = ['model_size', 'send_speed', 'send_time', 'data_size', 'compute_rate', 'learning_time', 'use_status']
learning_agent = []
learning_agent_name = []
agent_name = []

for index, dev in enumerate(device_name):
    # print(f'**{dev}의 Learning Agent')
    globals()[f'learning_agent_{dev}'] = []
    data = df_edge.at[dev, 'data_size']
    # print(data)
    for agent_num in range(opt.max_learning_agent_num):
        agent = device.learning_agent(opt.model_size, opt.max_learning_agent_speed, data)
        learning_agent.append(agent.item()) #learning agent spec 저장
        agent_name.append(f'{dev}_learning_agent_{agent_num+1}')
        # agent.info()
    learning_agent_name.append(f'learning_agent_{dev}')
# learning_agent = np.transpose(learning_agent)
# print(agent_name)
# print(learning_agent)
df_learning_agent = pd.DataFrame(learning_agent, columns=agent_index, index=agent_name)
# df_learning_agent = pd.DataFrame(learning_agent, columns=agent_name)
print('')
print('3. Learning Agent')
print(df_learning_agent)

# df = pd.concat([df_global,df_edge,df_learning_agent],index=agent_index)
df = pd.concat([df_global,df_edge,df_learning_agent],axis=0)
print('')
print('4. Total device Spec')
print(df)

df.to_csv('../' + opt.file_name_device + '.csv')

''' Round
1. 학습에 사용할 엣지 디바이스 선택
2. 선택된 엣지 디바이스에 모델 디플로이
3. 선택된 엣지 디바이스 로컬 학습
4. aggregation
'''

#output_index = []
#output_column = []
last_accuracy_list = []
old_accuracy_list = []
last_aggre_acc = 0
total_time_item = [] # 연합학습에서 소요된 총 시간
round_time_item = []
aggre_acc_list = []

#base_index = ['model deploy time', 'edge1_iter', 'edge1_learning_agent_1use_status', 'edge1_learning_agent_2use_status', 'edge1_send_time']

# edge2_iter
# edge2_learning_agent_1use_status
# edge2_learning_agent_2use_status
# edge2_send_time
# Learning Agent
# Learning Agent
# send_time
# Using Device
# Using Device
# send_time


df_round = pd.DataFrame()
for round in range(1, opt.round_num+1):

    ### round 마다 reset되는 정보 ###
    output_index = []  # 출력 파일 행 (round)
    output_column = [] # 출력 파일 열 (feature)
    round_item = []    # 정보
    accuracy_list = [] # 정확도
    round_time_list = []    # 시간
    # sum_acc = 0

    ### index(몇번째 round인지) ###
    round_name = 'Round' + str(round)
    output_index.append(round_name)
    print('')
    print(f'* Round {round}')

    ## 1. Edge device Select : round 참여 device 선택
    # device 선택
    print(f'  1. Edge device Select')
    print(f'     Total device: {device_name}')
    random_num = helper.random_integer(1, len(device_name))
    device_name_sample = random.sample(device_name, random_num)
    device_name_sample.sort()

    # device 정보(인덱스,이름) 추출 -> 인덱스가 왜 필요하지?
    target_device = [] # target_device : (index,device name)
    for g, device_g in enumerate(device_name):
        for device_s in device_name_sample:
            if device_s == device_g:
                target_device.append([g+1, device_s])
    print(f'     선택된 디바이스: {device_name_sample}')
    # print(f'     선택된 디바이스: {target_device}')

    ## 2. deploy : global model을 edge device에 전송 (소요 시간 = deploy_time)
    print(f'  2. Model deploy')
    deploy_time = global_model.time_s
    print(f'     Deploy time: {deploy_time}')
    output_column.append('model deploy time')
    round_item.append(deploy_time)

    ## 3. Select device local train : edge device에서 학습(소요 시간 = time + local train time)
    # Leaning Agent가 있을 경우 사용 가능 여부를 확인하여 학습을 옮겨서 실행행
    print(f'  3. Local train & Send to server ({len(device_name_sample)} device)')
    print(f'     Required time = learning time * iteration//10 + send time') # 왜 //10 하는지 확인

    #for index_s, sample in enumerate(target_device): # 랜덤하게 선택한 device의 인덱스와 이름
    for sample in target_device:
        # print(sample) # sample[0] : 인덱스 / sample[1] : sampling device name
        target_device_name = sample[1]
        max_iter_value = 4 # 후에 조정 - iteration 조정
        iteration = helper.random_integer(1, max_iter_value) * 50 # iteration
        print(f'       {sample[1]} iteration: {iteration}')

        output_column.append(target_device_name + '_iter')
        round_item.append(iteration)

        # learning agent 존재
        df_index = list(df.index)
        find = []
        la_time = 0
        la = 'None'
        for z in df_index:
            if sample[1]+'_' in z: # learning agent가 있을 경우
                find.append(z)
                use_status = df.loc[z].use_status

                output_column.append(z + 'use_status')
                round_item.append(use_status)

                if use_status == 1: # learning agent가 사용 가능한 경우
                    if la_time == 0:  # 사용가능한 learning agent가 없었던 경우
                        la_time = df.loc[z].send_time + df.loc[z].learning_time
                        la = z
                    else:  # 이전에 사용가능한 learning agent가 있던 경우
                        temp_la_time = df.loc[z].send_time + df.loc[z].learning_time
                        if la_time > temp_la_time:
                            la_time = temp_la_time
                            la = z
        # print('index:', df_index) # 전체 device
        # print('find: ', find)     # 사용 가능한 learning agent
        edge_time = df.loc[target_device_name].send_time + df.loc[target_device_name].learning_time
        print('       * edge_time : (edge device) send time + learning time')
        print('       * la_time : (learning agent) send time + learning time')
        print(f'       edge_time:{edge_time}')
        print(f'       learning agent:{la}, la_time:{la_time}')
        if edge_time <= la_time or la_time == 0:
            use_device = target_device_name
            send_time = edge_time
        else :
            use_device = la
            send_time = la_time
        round_time_list.append(send_time)
        # send time = send time + learning time
        print(f'       use_device:{use_device} / send_time:{send_time}',)

        # device accuracy
        # device name : use_device
        # accuracy = (data size * iteration) / f(=적절한 값)
        # f = data size + iteration
        f = df.loc[use_device].data_size + iteration
        # round1 : 이전 round의 accuracy가 존재하지 않음
        if round == 1 :
            accuracy = df.loc[use_device].data_size * iteration // f
            accuracy_list.append(accuracy)
            # print(df.loc[use_device].data_size)
            # print(iteration)
            # print(f)
            # print('acc: ', accuracy)
        # round2 이상 : 이전 round의 accuracy를 고려하여 accuracy 계산
        else :
            last_aggre_acc = aggre_acc_list[-1]
            accuracy = (df.loc[use_device].data_size * iteration // f) + last_aggre_acc
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
    aggre_f = 3*len(device_name) - len(device_name_sample) # device 개수가 많을수록 높은 accuracy가 계산되도록
    if round == 1:
        aggre_acc = sum(accuracy_list) // aggre_f
    else :
        aggre_acc = sum(accuracy_list) // aggre_f + aggre_acc_list[-1]
    aggre_acc_list.append(aggre_acc)
    print(f'     aggre_acc: {aggre_acc}')
    # print('aggre_acc_list: ', aggre_acc_list)

    # DataFrame 생성
    round_item = np.reshape(round_item, (1, len(round_item)))
    df_temp = pd.DataFrame(round_item, columns=output_column, index=output_index)
    #df_temp = df_temp.reset_index()
    print('')
    print(df_temp)
    df_round = pd.concat([df_round, df_temp], join='outer', axis=0)

df_round['round time'] = round_time_item
df_round['total round time'] = total_time_item
df_round['aggregation accuracy'] = aggre_acc_list
print(df_round)
print(f"연합학습 총 소요 시간 : {df_round['total round time'].sum()}")
print(f"연합학습 최종 Accuracy : {df_round['aggregation accuracy'][-1]}")
df_round.to_csv('../' + opt.file_name_round + '.csv')

# Accuracy Graph
X = list(df_round.index)
y = list(df_round['aggregation accuracy'])
plt.plot(X, y, linestyle='dashed', marker='.', color='violet')
plt.xlabel('Round')
plt.ylabel('Accuracy')
# plt.show()
plt.savefig('../' + opt.file_name_graph + '.png')