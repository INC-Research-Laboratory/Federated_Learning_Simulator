import random
import sys

def random_integer(start,end):
    return random.randint(start, end)

def model(model, size=None):
    if model=="MobileNet_V2":
        size = 14

    elif model=="Inception_V3":
        size = 95

    elif model=="Vgg19":
        size = 549

    elif model=="SRGAN":
        size = 6.1

    elif model=="mymodel":
        size = size

    return size

def dataset(dataset, volume=None, imgs=None, size=None):
    if dataset=="MNIST":
        volume = 66.6
        imgs = 60000
        size = (28,28,1)

    elif dataset=="CIFAR10":
        volume = 356.7
        imgs = 50000
        size = (32,32,3)

    elif dataset=="CelebA":
        volume = 1400
        imgs = 202599
        size = (218,178,3)

    elif dataset=="mydataset":
        volume = volume
        imgs = imgs
        size = tuple(size)

    return (volume, imgs, size)

def communication(comm, speed=None):
    if comm=="5G":
        speed = 800
    elif comm=="LTE":
        speed = 150
    elif comm=="Wifi":
        speed = 400
    elif comm=="lan_1G":
        speed = 1000
    elif comm=="lan_500M":
        speed = 500
    elif comm=="mycomm":
        speed = speed

    return speed

