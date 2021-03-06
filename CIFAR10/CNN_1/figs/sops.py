# encoding=utf-8
import matplotlib
matplotlib.rcParams['backend'] = 'SVG'
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from pylab import *                    
#mpl.rcParams['font.sans-serif'] = ['SimHei']
#simsun = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc', size=10) # simsun
roman = FontProperties(fname=r'C:\Windows\Fonts\times.ttf', size=15) # Times new roman
mpl.rcParams['font.sans-serif'] = ['SimSun']
#fontcn = {'family': 'SimSun','size': 10} # 1pt = 4/3px
fonten = {'family':'Times New Roman','size': 15}

sop0, sop1, sop2 = [[]], [[]], [[]]
batch_number, time_step = 0, 0
time_steps = 100
batch_numbers = 200
batch_size = 50
file_sop0 = open('.\\k0\\sop_num.txt')
file_sop1 = open('.\\k1\\sop_num.txt')
file_sop2 = open('.\\k2\\sop_num.txt')

for line in file_sop0.readlines():
    time_step = time_step + 1
    sop=int(line.strip('\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        sop0.append([sop])
    else:
        sop0[batch_number].append(sop)

file_sop0.close()
time_step, batch_number = 0, 0
sop0 = np.sum(np.array(sop0) / (batch_size*batch_numbers), axis=0)
sop0 = sop0[-1]

for line in file_sop1.readlines():
    time_step = time_step + 1
    sop=int(line.strip('\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        sop1.append([sop])
    else:
        sop1[batch_number].append(sop)

file_sop1.close()
time_step, batch_number = 0, 0
sop1 = np.sum(np.array(sop1) / (batch_size*batch_numbers), axis=0)
sop1 = sop1[-1]

for line in file_sop2.readlines():
    time_step = time_step + 1
    sop=int(line.strip('\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        sop2.append([sop])
    else:
        sop2[batch_number].append(sop)

file_sop2.close()
sop2 = np.sum(np.array(sop2) / (batch_size*batch_numbers), axis=0)
sop2 = sop2[-1]

## the number of MAC of CNN 1 on CIFAR10, a multiplication is supposed to be equivalent with two additions in this paper
mac0 = (256*24*24*96*3*3 + 256*12*12*256*2*2 + 384*12*12*256*3*3 + 384*6*6*384*2*2 + 384*6*6*384*3*3 + 256*6*6*384*3*3 + \
256*3*3*256*2*2 + 1024*1*1*256*3*3 + 1024*1024 + 1024*10)*3
mac1 = mac0
mac2 = mac0

# auto label
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))

name_list = ['k=0','k=1','k=2','ANN']  
num_list = [sop0,sop1,sop2, mac0]  
x =list(range(len(num_list)))  

autolabel(plt.bar(x, num_list, color='gybr', tick_label=name_list))
#plt.legend(prop=fonten)  

plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(" ", fontproperties=roman) 
plt.ylabel("SOPs", fontproperties=roman) 
plt.title("Computing operations on CIFAR10", fontproperties=roman) 

plt.show()

plt.savefig('scnn1_sop_cifar10.svg',format='svg')
