import random
import numpy as np
import sys

random.seed(40)

def random_integer(start,end):
    return random.randint(start, end)

def custom_sigmoid(x,a,b):
    # a = 정확도 첫 시작점
    # b = 정확도 상한선
    return a+(b-a)*(2*((1/(1+np.exp(-x)))-0.5))

