import random
import sys

def aggregation(list):
    # server에서 aggregation
    random_sample_num = random.randint(1,len(list))
    sample_device_name = random.sample(list, random_sample_num)
    aggregation_delay = 5
    aggregation_time = len(sample_device_name) * aggregation_delay
    print(f'  참여 device : {sample_device_name}')
    print(f'  aggregation_delay : {aggregation_delay}')
    print(f'  aggregation_time : {aggregation_time}')
    return aggregation_time

def random_integer(start,end):
    return random.randint(start, end)

def make_input():
    # 입력 함수
    print('입력 내용 : global model size,server speed,edge speed, learning agent speed')
    n = list(map(int, sys.stdin.readline().split(sep=',')))
    global_model = n[0]
    server_speed = n[1]
    edge_speed = n[2]
    learning_agent_speed = n[3]
    print(n[0])


