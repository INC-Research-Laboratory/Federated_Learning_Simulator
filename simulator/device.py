import helper

class server:
    state = 'server'

    def __init__(self, model_size, send_speed):
        self.model_size = model_size
        self.send_speed = send_speed
        self.time_s = model_size / send_speed

    def info(self):
        print(f'-------{self.state} 생성-------')
        attribute = {'모델 사이즈':self.model_size, '전송 속도':self.send_speed, '전송 시간':self.time_s}
        for attribute_name, attribute_value in attribute.items():
            print(f'{attribute_name:10s}{attribute_value:.2f}')

    def item(self):
        item = [self.model_size, self.send_speed, self.time_s]
        return item

class edge(server):
    state = 'edge'
    # 연산 성능 random 필요
    def __init__(self, model_size, send_speed, edge_data):
        super().__init__(model_size, send_speed)
        self.edge_send_speed = helper.random_integer(send_speed//2, send_speed)
        self.send_speed = self.edge_send_speed
        self.data = helper.random_integer(edge_data, edge_data*2) * 10
        self.compute_rate = 100
        self.learning_time = self.data/self.compute_rate
        self.time_s = model_size / self.edge_send_speed

    def info(self):
        super().info()
        attribute = {'data 사이즈': self.data, '연산 성능':self.compute_rate, '학습 시간': self.learning_time}
        for attribute_name, attribute_value in attribute.items():
            print(f'{attribute_name:10s}{attribute_value:.2f}')

    def learning(self):
        update_time_s = self.time_s + self.learning_time
        print(f'{update_time_s:.2f}(학습 후 전송시간)={self.time_s:.2f}(기존 전송시간)+{self.learning_time:.2f}(학습시간)')

    def item(self):
        item = [self.model_size, self.send_speed, self.time_s, self.data, self.compute_rate, self.learning_time]
        return item

class learning_agent(edge):
    state = 'learning agent'

    def __init__(self, model_size, send_speed, edge_data):
        super().__init__(model_size, send_speed, edge_data)
        self.send_speed = send_speed
        self.data = edge_data
        self.compute_rate = helper.random_integer(3, 6) * 50
        self.learning_time = self.data / self.compute_rate
        self.use_status = helper.random_integer(0,1) #0=false / 1=true
        self.time_s = model_size / send_speed

    def info(self):
        super().info()
        attribute = {'사용여부':self.use_status}
        for attribute_name, attribute_value in attribute.items():
            print(f'{attribute_name:10s}{attribute_value:.2f}')
    def item(self):
        item = [self.model_size, self.send_speed, self.time_s, self.data, self.compute_rate, self.learning_time, self.use_status]
        return item
if __name__ == "__main__":

    global_model = server(20,10)
    global_model.info()
    edge1 = edge(global_model.model_size, helper.random_integer(1,10), 40)
    edge1.info()
    edge1_la = learning_agent(global_model.model_size, helper.random_integer(10,20), edge1.data)
    edge1_la.info()
    '''
    la = learning_agent(10, 5, 30)
    print(la.__dict__)
    la.info()
    '''