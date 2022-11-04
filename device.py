import pandas as pd

import helper
import numpy as np

class server:
    state = 'server'

    def __init__(self, model_size, communication):
        self.model_size = model_size
        self.communication = communication/8
        self.time = round(model_size / self.communication, 3)

    def info(self):
        print(f'-------{self.state} 생성-------')
        attribute = {'모델 사이즈':self.model_size, '통신':self.communication, '전송 시간':self.time}
        for attribute_name, attribute_value in attribute.items():
            print(f'{attribute_name:10s}{attribute_value:.2f}')

    def item(self):
        item = [self.model_size, self.communication, self.time]
        return item

    def df_global(self, list):
        global_shape = np.shape(list)
        global_item_list = np.reshape(list,(1, global_shape[0]))
        # global_item_list = np.transpose(global_item_list)
        global_index = ['Model_Size', 'Communication_Rate', 'Transmission_Delay']
        df_global = pd.DataFrame(global_item_list, columns=global_index, index=['global'])
        # df_global = pd.DataFrame(list, columns=global_index, index=['global'])
        return df_global

class edge(server):
    state = 'edge'
    # 연산 성능 random 필요
    def __init__(self, model_size, communication, dataset, compute_rate):
        super().__init__(model_size, communication)
        self.communication = communication/8
        self.dataset = dataset
        self.compute_rate = compute_rate
        self.learning_time = self.dataset * self.compute_rate # ms -> s
        self.time = round(model_size / self.communication, 3)

    def info(self):
        super().info()
        attribute = {'data 사이즈': self.dataset, 'device 성능':self.compute_rate, '학습 시간': self.learning_time}
        for attribute_name, attribute_value in attribute.items():
            print(f'{attribute_name:10s}{attribute_value:.2f}')

    def item(self):
        item = [self.model_size, self.communication, self.time, self.dataset, self.compute_rate, self.learning_time]
        return item

    def df_edge(self, list, name):
        edge_index = ['Model_Size', 'Communication_Rate', 'Transmission_Delay', 'Dataset_Size', 'Computing_Power', 'Learning_Time']
        df_edge = pd.DataFrame(list, columns=edge_index, index=name)
        return df_edge

class learning_agent(edge):
    state = 'learning agent'

    def __init__(self, model_size, communication, dataset, compute_rate):
        super().__init__(model_size, communication, dataset, compute_rate)
        self.communication = communication/8
        self.dataset = dataset
        self.compute_rate = compute_rate
        self.learning_time = self.dataset * self.compute_rate
        self.use_status = helper.random_integer(0,1) #0=false / 1=true
        self.time = round(model_size / self.communication, 3)


    def info(self):
        super().info()
        attribute = {'사용여부':self.use_status}
        for attribute_name, attribute_value in attribute.items():
            print(f'{attribute_name:10s}{attribute_value:.2f}')
    def item(self):
        item = [self.model_size, self.communication, self.time, self.dataset, self.compute_rate, self.learning_time, self.use_status]
        return item

    def df_agent(self, list, name):
        agent_index = ['Model_Size', 'Communication_Rate', 'Transmission_Delay', 'Dataset_Size', 'Computing_Power', 'Learning_Time', 'Use_Status']
        df_agent = pd.DataFrame(list, columns=agent_index, index=name)
        return df_agent