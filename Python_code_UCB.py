import csv
import np as np
import pandas as pd
import numpy as np
import math
import random
from numpy import genfromtxt

np.set_printoptions(threshold=np.inf)
#English Edition
# pseudo-code of UCB
# FIRSTLY, we need t = 1:504 of J matrix to train our UCB
# for t = 1：504
#   reward = j[t,:]
#   for action = 1:4096
#       if reward[action]==-inf
#           Q[action] = 3/5 Q[action]
#           by observaing，reward of most action becomes -inf at some timeslot. so once -inf occurs,
#           it is not mean this choice is not good for all time slots, we need a parameter 3/5 (or other value)
#            to reduce Q of this action, but we should not cut down it to -inf
#
#       if reward[action] != -inf
#           Q[action] = Q[action] + lr*(reward[action] - Q[action])
# SECONDLY,test
# for t = 505:1008
#   execute UCB
#
######################################################################################################
# Chinese Edition
# 伪代码 标准版UCB
# 取J的前504个时隙的数据
# 一：训练
# for t = 1：504
#   reward = j[t,:]
#   for action = 1:4096
#       如果reward[action]==-inf
#           Q[action] = 3/5 Q[action] 通过观察，J表中大部分action都会在某个时间出现-inf，不能因为出现-inf就认为这个选项不好。当前Q是之前Q的五分之三，用递减来达到逐渐减小Q的目的。这样，如果该action连续为-inf，Q就会很小。
#           备用选择:Q[action] = Q[action] - lr*Q[action]这里把R记为0，效果和上面的差不多
#       如果reward[action] != -inf
#           Q[action] = Q[action] + lr*(reward[action] - Q[action])新Q = 老Q + 估计误差
# 二：检验
# for t = 505：1008
#   使用Q表做 At = argmax（Qt + c....）
#   计算R[At] = J[t, At]
#   更新N的值
#   更新Q[At] = Q[action] - lr*(R[At] - Q[action])
#




class Q_learning:
    def __init__(self):
        self.bandits = pow(2, 12)
        self.timeslots = 1008
        self.J = np.array((1009, self.bandits + 1))
        self.Q_UCB = np.zeros(self.bandits + 1)
        self.lr = 0.99
        self.N = np.zeros(self.bandits + 1)
        self.ATt = np.zeros((1010, 4097))
        self.num_sc = 12
        self.c = 2
        self.parameter_a = 1 / 5
        self.R_training = np.zeros(self.bandits + 1)
        self.Q_training = np.zeros(self.bandits + 1)

    def best_result(self):
        self.T_opt = np.max(self.J, 1);

    def generate_table(self):
        self.J = genfromtxt(r'C:\Users\yanglianrui\Desktop\J_value.csv', delimiter=',');

    def training(self):  # 成功 success
        self.generate_table()
        for t in range(1, 504):
            reward = self.J[t, :]
            # print(reward)
            for act in range(1, self.bandits + 1):
                if reward[act] == -float('inf'):
                    self.Q_training[act] = self.parameter_a * self.Q_training[act]

                else:
                    self.Q_training[act] = self.Q_training[act] + self.lr * (reward[act] - self.Q_training[act])
        self.N[1:] = 504
        self.ATt[504, :] = self.Q_training + self.c * math.sqrt(math.log(504) / 504)
        print(self.Q_training)  # see Q_trainingm table
        a = np.max(self.Q_training)
        b = np.argmax(self.Q_training)
        print(a, b)  # see the max value of Q_training table and its argument

    def UCB(self):
        self.generate_table()
        self.Q_UCB = self.Q_training
        self.best_result()
        for t in range(505, self.timeslots + 1):

            action = np.argmax(self.ATt[t, :])
            R = self.J[t, action]
            self.N[action] += 1
            if R == -float('inf'):
                self.Q_UCB[action] = self.parameter_a * self.Q_UCB[action]
            else:
                self.Q_UCB[action] = self.Q_UCB[action] + self.lr * (R - self.Q_UCB[action])
            for act in range(1, self.bandits + 1):
                self.ATt[t + 1, act] = self.Q_UCB[act] + self.c * math.sqrt(math.log(t) / self.N[act])
            print("time,action,reward,       Percentage Compared with Reward of the Best choice(PCRB)")
            print(t, action, R,R / self.T_opt[t])


A = Q_learning()
A.training()
A.UCB()
# CONCLUSION:
# The performance is not always good. In fact, compared with the best reward at each timeslot
# the reward of UCB (PCRB) is around 50% - 80%. Rarely, -inf occurs. Sometimes, the PCRB is lower
# than 15%. The total performance depends on the value of self.c and self.parameter_a. 
# We can say that this algorithm can save energy mean while maximize revenue.
