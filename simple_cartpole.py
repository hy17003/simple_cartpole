#coding=utf-8

import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

'''
环境类
'''
class env:
    def __init__(self):
        self.game = gym.make('CartPole-v0')
    '''
    环境执行一次动作，返回反馈信息
    '''
    def step(self,action):
        state, reward, done, _ = self.game.step(action)
        return state, reward, done


'''
智能体类
'''
class agent:
    def __init__(self):
        self.buffer = deque()
        self.max_buffer = 10000
        self.state_dim = 4
        self.action_dim = 2
        self.epsilon = 0.2

    '''
    获得一次动作，该动作有一定的随机性
    '''
    def get_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

    '''
    感知环境，保存游戏返回的状态及奖励等信息
    '''
    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.buffer.append(state, one_hot_action, reward, next_state, done)
        if len(self.buffer) > self.max_buffer:
            self.buffer.popleft()
    '''
    创建Q值网络，输入为当前状态state_input
    '''
    def create_Q_network(self):
        W1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        self.Q_value = tf.matmul(h_layer, W2) + b2
    '''
    训练Q值网络
    '''
    def train_Q_network(self):
        return 0
    '''
    初始化权值
    '''
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)
    '''
    初始化偏置值
    '''
    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)


'''
主函数
'''
def main():
    return 0







