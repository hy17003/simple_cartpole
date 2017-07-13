#coding=utf-8

import gym
import tensorflow as tf
import numpy as np
import random
import os
from collections import deque

'''
环境类
'''


class Environment:
    def __init__(self, sess):
        self.game = gym.make('CartPole-v0')
        self.session = sess

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

    def __init__(self, sess):
        self.buffer = deque()
        self.max_buffer = 10000
        self.state_dim = 4
        self.action_dim = 2
        self.epsilon = 0.1
        self.batch_size = 10
        self.session = sess
        self.create_q_network()
        self.create_train_method()
        self.saver = tf.train.Saver()
        self.start_idx = 0


    '''
    决策，根据状态计算Q值，选择最大Q值的动作输出，同时，该动作有一定的随机性
    '''

    def get_greedy_action(self, state):
        q_value = self.session.run(self.q_value, feed_dict={self.state_input: [state]})
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(q_value[0])

    '''
    决策，根据状态计算Q值，选择最大Q值的动作输出
    '''
    def get_action(self, state):
        q_value = self.session.run(self.q_value, feed_dict={self.state_input: [state]})
        return np.argmax(q_value[0])

    '''
    感知环境，保存游戏返回的状态及奖励等信息
    '''

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.buffer) > self.max_buffer:
            self.buffer.popleft()
        if len(self.buffer) > self.batch_size:
            self.train_q_network()

    '''
    创建Q值网络，输入为当前状态state_input, 网络输出两个动作的Q值
    '''

    def create_q_network(self):
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        W1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        self.q_value = tf.matmul(h_layer, W2) + b2

    '''
    创建训练方法，输入为动作，期望奖励及当前状态，Q值网络根据当前状态估计两个动作Q值，获取当前动作对应的Q值，该Q值与
    输入期望奖励差的平方即为cost，因为在网络稳定时，因为网络稳定后，Q值表中，每一个(状态，动作)上的值将不变
    '''
    def create_train_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None, ])
        q_action = tf.reduce_sum(tf.multiply(self.q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - q_action))
        self.train_step = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    '''
    训练Q值网络
    '''
    def train_q_network(self):
        minibatch = random.sample(self.buffer, self.batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        y_batch = []
        for i in range(self.batch_size):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                reward_ex = self.session.run(self.q_value, feed_dict={self.state_input: [next_state_batch[i]]})
                y_batch.append(reward_batch[i] + np.max(reward_ex))
        self.session.run(self.train_step, feed_dict = {self.state_input: state_batch, self.y_input:y_batch,
                                                       self.action_input:action_batch})

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
    保存网络模型
    '''
    def save_model(self, path, step):
        model_name = os.path.join(path, 'cartpole.ckpt')
        self.saver.save(self.session, model_name, global_step=step)
        return 0

    '''
    载入网络模型
    '''
    def load_model(self, path):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            self.start_idx=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

'''
主函数
'''
MAX_TIMES = 100000
MAX_STEP = 300
def main(stage):
    sess = tf.Session()
    env = Environment(sess)
    agent = Agent(sess)
    sess.run(tf.global_variables_initializer())
    agent.load_model('model')
    if(stage == 'TRAIN'):
        for t in range(int(agent.start_idx), MAX_TIMES):
            cur_state = env.restart()
            score = 0
            for step in range(MAX_STEP):
                #根据当前状态，智能体作出一种动作
                action = agent.get_greedy_action(cur_state)
                #执行这种动作，获得环境反馈
                next_state, reward, done = env.step(action)
                score = score + reward
                # 显示画面
                if(t > 3000):
                    env.show()
                #智能体感知这些反馈
                agent.perceive(cur_state, action, reward, next_state, done)
                #更新环境状态
                cur_state = next_state
                if done:
                    break;
            print("t = ", t, " Score = ", score)
            if t % 2000 == 0:
                agent.save_model('model', t)
    else:
        for t in range(100):
            cur_state = env.restart()
            score = 0
            for step in range(MAX_STEP):
                # 根据当前状态，智能体作出一种动作
                action = agent.get_action(cur_state)
                # 执行这种动作，获得环境反馈
                next_state, reward, done = env.step(action)
                score = score + reward
                # 显示画面
                env.show()
                # 智能体感知这些反馈
                agent.perceive(cur_state, action, reward, next_state, done)
                # 更新环境状态
                cur_state = next_state
                if done:
                    break;
            print("t = ", t, " Score = ", score)
    sess.close()

if __name__ == '__main__':
  main('TEST')
