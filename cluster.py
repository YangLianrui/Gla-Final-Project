"""
伪代码
创建对象 BaseStation:
    def __init__(self):
        self.




拿到所有small BSs的数据demandData_small_BSs[]
拿到Macro BS的数据demandData_macro_BS
计算demandData_small_BSs[]的平均值



对象中应该包括 BS标号（1、2、3、4...）、属性（Macro BS/ Small BS）、数值（float(demandData)）、分属的cluster （二分类 使用10表示，注意Small BSs的个数和10的位数对应）、
开关状态（On/Off或1/0）、能耗（float）、时间（t）


"""

import csv
import np as np
import pandas as pd
import numpy as np
import math
import random
from numpy import genfromtxt
np.set_printoptions(threshold=np.inf)

class cluster:
    def __init__(self, value, time, mark):
        self.time = time
        self.mark = mark
        self.BSValue = value




# class cluster:
#     def __init__(self):
#         self.BaseStation = np.array(1009.4097)
#
#     def generateR(self):
#         # 生成 R R里包括[时间, 列]，traffic demand
#
#
#     def generateEneC(self):
#         # 生成能量表energy_C,包括[时间，列]，里面有能量消耗数据
#
#     def execution(self):
#         #执行generate R
#         #执行generateEneC
#         for timeslots in range(1, 1008):
#             self.BaseSsation[timeslots, 1] =  BS('Macro', 1, R[timeslots,1], 'N', 1, energy_C[timeslots, 1], timeslots)
#             # 在每个timeslot中生成MacroBS、FemoBS。。等的数据
#             # 计算MBS的1-R
#


            # 按照R的值，给除MacroBS外的所有BS从大到小排序
            # 计算除MBS外的所有BS的中位数
            # 按照中位数将BS分为两类
            # 找出较小的那一类，给类中的所有BS标记+1
            # 将较小一类的 R 求和
            # 将上述求和值和MBS 1-R 比较大小
            # 如果求和大于MBS 1-R， 重复。如果小于，给类中的所有BS标记 Off


            # 计算所有Off的BS的energy_C的和。
            # 总耗能 - energy_C的和 = 节约能量

R_load = genfromtxt(r'C:\Users\yanglianrui\Desktop\milan_data_L.csv', delimiter=',');
R_insert_col = np.zeros(102, dtype=float);
R_insert_row = np.zeros(3024, dtype=float);
R1 = np.array(R_load);
R2 = np.c_[R_insert_row.T, R1];  # add zeros to column 0
R = np.row_stack((R_insert_col, R2));  # add zeros to row 0
Macro_traffic = np.array(1 - R[:, 1])
BS_Offload1 = np.array(1 - R[:,:])
BS_Offload = np.delete(BS_Offload1, 1, axis = 1)
BS_small = []
Total_Off_set = np.zeros((3025, 75))

for i in range(1, 3025):
    BS_small = []
    print("heiheihei")
    # 把cluster对象放进list中
    for j in range(1, 101):
        s = cluster(BS_Offload[i, j], i, j)
        BS_small.append(s)
    # 排序部分
    for k in range(0, 100):
        for x in range(100 - k -1):
            if BS_small[x].BSValue > BS_small[x+1].BSValue:
               BS_small[x], BS_small[x+1] = BS_small[x+1], BS_small[x]
    #######
    # 找出比macro大的，并把arg记录到on_set中。on的意思是：开，在这个set中的BS不能关闭。
    on_set = []
    for z in range(len(BS_small)):
        #print(z)
        if BS_small[z].BSValue > Macro_traffic[i]:
            on_set.append(z)

    # 根据on_set剔除BS
    BS_small = [BS_small[q] for q in range(0, len(BS_small), 1) if q not in on_set]
    for v in range(len(BS_small)):
        print(BS_small[v].mark)
    # 第一次比对和
    sum_1 = 0
    for q in range(len(BS_small)):
        sum_1 = sum_1 + BS_small[q].BSValue
    # 如果减去BS_small最大值后的sum_1比Macro大，则记录减去的这个BS_small在on_set_1中。循环
    on_set_1 =[]
    for q in reversed(range(len(BS_small))):
        sum_1 = sum_1 - BS_small[q].BSValue
        on_set_1.append(BS_small[q])


        if  sum_1 <= Macro_traffic[i] :
            break


    # 把on_set_1中的元素剔除，剩下可以使用的
    for BBB in on_set_1:

        BS_small.remove(BBB)

    for AAA in range(len(BS_small)):
        Total_Off_set[i , AAA] = BS_small[AAA].mark
print(Total_Off_set)