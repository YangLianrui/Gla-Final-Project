import csv
import np as np
import pandas as pd
import numpy as np
import math
import random
from numpy import genfromtxt
np.set_printoptions(threshold=np.inf)
# 解释文档
class Q_learning:
    def __init__(self,num_sc=12, epsilon=0.1):
        self.J = np.array((1009,4096))
        self.R = 0;
        self.num_sc = num_sc;
        self.TimeSlots = 1008;
        self.R_list = []

        self.N_arg = np.zeros(4096)
        self.T_opt = np.array(1009)

        self.Q_e_greedy = np.zeros(pow(2, num_sc))



    # useless
    def generate_table(self):
        self.J = genfromtxt(r'C:\Users\yanglianrui\Desktop\J_value.csv', delimiter=',');

    def best_result(self):
        self.T_opt = np.max(self.J, 1);



    def get_rewards(self, arg_time_slot_j, arg_sc_i):
        return self.J[arg_time_slot_j, arg_sc_i + 1];
    # get_rewards from J，where j is the argument of timeslot，i is one of 4096 combinations


    def e_greedy(self,epsilon):
        self.best_result()
        alph = 0.5
        for j in range(1, self.TimeSlots + 1):#j是time slot角标 # j is time slot
            if np.random.random() > epsilon:# 如果比epsilon大，则选取已知最好的sc组合 # if ramdom is larger that epsilon, choose the best choice the we current know
                if j == 1:
                    arg_sc_i = np.random.randint(1,pow(2,self.num_sc))
                else:
                    arg_sc_i = np.argmax(self.Q_e_greedy) + 1
            else:# 如果比epsilon小，则随机选择sc组合
                arg_sc_i = np.random.randint(1,pow(2,self.num_sc))
            reward = self.get_rewards(j, arg_sc_i);
            print("e-greedy")
            print(arg_sc_i)
            print(reward)
            a = reward / self.T_opt[j]
            print(a)
            self.N_arg[arg_sc_i - 1] += int(1);

                #self.R += (reward - self.R) / j
            self.R = reward
                # for A in range(1, int(self.N_arg[arg_sc_i - 1])): # 计算Q(n+1) AAA是一个表，sum(i=1:n) a(1-a)^(n-i)R(i) n是该arm已经被选过的次数
                #     self.AAA[A] = alph * pow((1 - alph), (self.N_arg[arg_sc_i - 1] - A)) * self.R

            if self.Q_e_greedy[arg_sc_i - 1] == -float('inf'):
                self.Q_e_greedy[arg_sc_i - 1] = -float('inf')
                # print("Q")
                # print(arg_sc_i)
                # print(self.Q_e_greedy[arg_sc_i - 1])
            elif self.R == -float('inf'):
                self.Q_e_greedy[arg_sc_i - 1] = -float('inf')
                # print("Q")
                # print(arg_sc_i)
                # print(self.Q_e_greedy[arg_sc_i - 1])
            else:
                    self.Q_e_greedy[arg_sc_i - 1] = self.Q_e_greedy[arg_sc_i - 1] + (np.array(self.R) - self.Q_e_greedy[arg_sc_i - 1]) / self.N_arg[arg_sc_i - 1]
                    # print("Q")
                    # print(arg_sc_i)
                    # print(self.Q_e_greedy[arg_sc_i - 1])
                #self.Q_e_greedy[arg_sc_i - 1] = pow((1 - alph), self.N_arg[arg_sc_i - 1]) * self.Q_e_greedy[arg_sc_i - 1] + np.sum(self.AAA)
            #print(self.R)

    def printQ(self): # print the Q table of e-greedy
        print(self.Q_e_greedy)

    def random(self): # randomly choose arms
        self.best_result()
        for j in range(1, self.TimeSlots + 1):
            arg_sc_i = np.random.randint(1, pow(2, self.num_sc))
            reward= self.get_rewards(j, arg_sc_i)
            print("random")
            print(reward)
            print(reward / self.T_opt[j])
    # def generate_argmaxTable(self,N_times ,Q_table ):
    #     argmaxTable = self.num_sc
    # def UCB(self):
    #     for t in range(1, self.TimeSlots + 1):
    #
    #         At = np.argmax()



A = Q_learning()
A.generate_table()
A.e_greedy(0.5)
A.random()
A.printQ()

