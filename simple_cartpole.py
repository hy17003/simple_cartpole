#coding=utf-8

import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

'''
环境类
'''


class Environment:
    def __init__(self):
        self.game = gym.make('CartPole-v0')

    '''
    环境执行一次动作，返回反馈信息
    '''
    def step(self,action):
        state, reward, done, _ = self.game.step(action)
        return state, reward, done

    '''
    显示画面
    '''
    def show(self):
        self.game.render()

    '''
    重启
    '''
    def restart(self):
        return self.game.reset()

'''
智能体类
'''


class Agent:
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
    创建Q值网络，输入为当前状态state_input, 网络输出两个动作的Q值
    '''
    def create_Q_network(self):
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        W1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        self.Q_value = tf.matmul(h_layer, W2) + b2

     '''
     创建训练方法，输入为动作，期望奖励及当前状态，Q值网络根据当前状态估计两个动作Q值，获取当前动作对应的Q值，该Q值与
     输入期望奖励差的平方即为cost，因为在网络稳定时，因为网络稳定后，Q值表中，每一个(状态，动作)上的值将不变
     '''
    def create_train_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None, self])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.train_step = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def train_Q_network(self):
        return 0


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
MAX_TIMES = 100000
MAX_STEP = 300
def main():
    env = Environment()
    agent = Agent()
    for t in range(MAX_TIMES):
        cur_state = env.restart()
        for step in range(MAX_STEP):
            #根据当前状态，智能体作出一种动作
            action = agent.get_action(cur_state)
            #执行这种动作，获得环境反馈
            next_state, reward, done = env.step(action)
            # 显示画面
            env.show()
            #智能体感知这些反馈
            agent.perceive(cur_state, action, reward, next_state)
            #更新环境状态
            cur_state = next_state
            if done:
                break;
    return 0







