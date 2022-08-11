import argparse
import pandas as pd
import numpy as np
import random

import device
import helper

parser = argparse.ArgumentParser()
parser.add_argument("--file_name_device", type=str, default="device_spec", help="name of csv file")
parser.add_argument("--edge_num", type=int, default=2, help="number of edge device")
parser.add_argument("--round_num", type=int, default=1, help="number of round")
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
total_time = 0 # 연합학습에서 소요된 총 시간
last_accuracy_list = []
old_accuracy_list = []
last_aggre_acc = 0

for round in range(1, opt.round_num+1):
    print('')
    print(f'* Round {round}')
    accuracy_list = []
    sum_acc = 0
    step2_time = []

    # 1. Edge device Select
    print(f'  1. Edge device Select')
    print(f'     Total device: {device_name}')
    random_num = helper.random_integer(1, len(device_name))
    device_name_sample = random.sample(device_name, random_num)
    device_name_sample.sort()
    print(f'     선택된 디바이스: {device_name_sample}')

    # 2. deploy : time = deploy delay
    print(f'  2. Model deploy')
    deploy_time = global_model.time_s
    print(f'     Deploy time: {deploy_time}')

    # 3. Select device local train : time = time + local train time
    # Leaning Agent가 있을 경우 사용 가능 여부를 확인하여 학습을 옮겨서 실행행
    print(f'  3. Local train & Send to server ({len(device_name_sample)} device)')
    print(f'     Required time = learning time * iteration//10 + send time')

    target_device = []
    for g, device_g in enumerate(device_name): # 선택된 device 정보를 target_device에 저장
        for device_s in device_name_sample:
            if device_s == device_g:
                target_device.append([g+1, device_s])



    for index_s, sample in enumerate(target_device): # 랜덤하게 선택한 device의 인덱스와 이름
        # print(sample) # sample[0] : 인덱스 / sample[1] : sampling device name
        max_iter_value = 4 # 후에 조정 - iteration 조정
        iteration = helper.random_integer(1, max_iter_value) * 50
        print(f'       {sample[1]} iteration: {iteration}')
        # learning agent 존재
        temp = [0,0]
        #print(df_learning_agent.loc[target_device[1]])

#print(df_learning_agent.loc[['edge1_learning_agent_1','edge1_learning_agent_2']])
#print(device_name_sample)
#print(df_learning_agent.loc[[device_name_sample]])