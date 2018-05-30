import numpy as np
import tensorflow as tf

# 加载matplotlib工具包,使用该工具可以对预测的sin函数曲线进行绘图
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.pyplot import as plt

HIDDEN_SIZE = 30
NUM_LAYERS = 2

TIMESTEPS = 10              # 循环神经网络的训练序列长度
TRAINING_SETEPS = 10000     # 训练轮数
BATCH_SIZE = 32             # batch的大小

TRAINING_EXAMPLES = 10000   # 训练数据的个数
TESTING_EXAMPLES = 1000     # 测试数据的个数
SAMPLE_GAP = 0.01           # 采样间隔

def generate_data(seq):
    X = []
    y = []
    pass