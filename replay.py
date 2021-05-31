import tensorflow as tf 
import numpy as np

import math
import random

from utils.generic_utils import load_dataset_at
from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST

class Generate_sample():
    def __init__(self, dataset_index, size=200000, max_length=200): # 50000
        self.num_var = MAX_NB_VARIABLES[dataset_index]
        self.max_actions = NB_CLASSES_LIST[dataset_index] + 1  # 加上等待的动作
        self.labels = NB_CLASSES_LIST[dataset_index]  # 数据集label的数量
        self.max_steps = MAX_TIMESTEPS_LIST[dataset_index]
        self.dataset_set = load_dataset_at(dataset_index)[0]
        self.dataset_label = load_dataset_at(dataset_index)[1]
        self.len_partial_series0 = np.zeros([size])  # 记录一条状态的长度
        self.reward0 = np.zeros([size])
        self.actions0 = np.zeros([size])
        self.done0 = np.zeros([size])
        self.time_series_index0 = np.zeros([size])
        self.count0 = 0

        self.len_partial_series1 = np.zeros([size])  # 记录一条状态的长度
        self.reward1 = np.zeros([size])
        self.actions1 = np.zeros([size])
        self.done1 = np.zeros([size])
        self.time_series_index1 = np.zeros([size])
        self.count1 = 0

        self.size = size
        # self.learning_step = self.count
        # self.learning_step = 0

        self.lam = 0.001
        self.p = 1/3
        self.t = 0

        # self.generate()

    def cal_reward(self, action, current_data_index):
        reward = 0
        # print("action range: ", range(self.max_actions))
        if action == self.labels:  # 动作为等待
            reward = -self.lam * math.pow(self.t, self.p)
            # reward = 0
        elif action == self.dataset_label[current_data_index]:
            # reward = 1
            reward = math.sqrt(self.t/197)
        else:
            # reward = -1
            reward = math.sqrt(self.t/197) - 1
        return reward

    def generate(self):
        done = False
        index_scal = self.dataset_set.shape[0]
        length_scal = self.dataset_set[0].shape[1]
        for i in range(index_scal):  # i表示当前序列的index  (100, 2, 147)
            for j in range(1,length_scal):  # j表示当前信息在序列i中的位置（t）
                self.t = j
                # print("self_t: ", j)
                for m in range(self.max_actions):  # m遍历0，1，2三个动作
                    if m in range(self.labels):
                        done = True
                    else:
                        done = False
                    self.store_transition(i, j, m, self.cal_reward(m,i), done)
        # print("self.count: ", self.count)

    def store_transition(self, current_data_index, s, action, reward, done):
        if self.dataset_label[current_data_index] == 0:
            self.time_series_index0[self.count0] = current_data_index  # 存入当前transition所属的时间序列序号
            self.len_partial_series0[self.count0] = s
            # print("len partial series: ", self.len_partial_series[self.count], "count: ", self.count)
            self.actions0[self.count0] = action
            self.reward0[self.count0] = reward
            # print("reward: ", reward)
            self.done0[self.count0] = done
            self.count0 += 1
            if self.count0 >= self.size:
                self.count0 = 0
        elif self.dataset_label[current_data_index] == 1:
            self.time_series_index1[self.count1] = current_data_index  # 存入当前transition所属的时间序列序号
            self.len_partial_series1[self.count1] = s
            # print("len partial series: ", self.len_partial_series[self.count], "count: ", self.count)
            self.actions1[self.count1] = action
            self.reward1[self.count1] = reward
            # print("reward: ", reward)
            self.done1[self.count1] = done
            self.count1 += 1
            if self.count1 >= self.size:
                self.count1 = 0
        # self.learning_step = self.count - 1
    def sample(self, batch_size):
        assert batch_size <= self.count0
        assert batch_size <= self.count1
        rand_prob =np.random.uniform(low=0.0, high=1.0, size=None)
        if self.count1 == 0 or rand_prob < 0.88:  # 0.9:  # 0.5:
            idxes = random.sample(range(self.count0), batch_size)[0]
            action = self.actions0[idxes]
            done = self.done0[idxes]
            reward = self.reward0[idxes]
            label = 0
            time_point = self.len_partial_series0[idxes].astype(np.int32)  # 获取sample的transition所含的partial序列的位置
            print("time_point: ", time_point)
            serie_index = self.time_series_index0[idxes].astype(np.int32)  # 获取sample的transition所属的时间序列序号
            print("serie_index: ", serie_index)
            s = np.transpose(self.dataset_set, (0,2,1))[serie_index][:time_point]
            # print(s)
            if action in range(self.labels):
                s_ = np.transpose(self.dataset_set, (0,2,1))[serie_index][:time_point]
            else:
                s_ = np.transpose(self.dataset_set, (0,2,1))[serie_index][:time_point+1] 
        else:
            idxes = random.sample(range(self.count1), batch_size)[0]
            action = self.actions1[idxes]
            done = self.done1[idxes]
            reward = self.reward1[idxes]
            label = 1
            time_point = self.len_partial_series1[idxes].astype(np.int32)  # 获取sample的transition所含的partial序列的位置
            print("time_point: ", time_point)
            serie_index = self.time_series_index1[idxes].astype(np.int32)  # 获取sample的transition所属的时间序列序号
            print("serie_index: ", serie_index)
            s = np.transpose(self.dataset_set, (0,2,1))[serie_index][:time_point]
            # print(s)
            if action in range(self.labels):
                s_ = np.transpose(self.dataset_set, (0,2,1))[serie_index][:time_point]
            else:
                s_ = np.transpose(self.dataset_set, (0,2,1))[serie_index][:time_point+1] 
        print("label: ", label)
        # self.learning_step -= 1
        return s, \
               action, \
               reward, \
               s_, \
               done 

if __name__ == "__main__":
    generator = Generate_sample(dataset_index=38)
    generator.generate()
# # #     # for i in generator.len_partial_series:
# # #     #     print(i)
# # #     print(len(generator.len_partial_series))
    # print(generator.dataset_label)
    for i in range(5):
        s, action, reward, s_, done = generator.sample(1)
        print('s: ', s)
        print('action: ', action)
        print('reward: ', reward)
        print('s_: ', s_)
        print('done: ', done)
        # print('t: ', generator.learning_step)