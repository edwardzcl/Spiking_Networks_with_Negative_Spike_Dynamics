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

spike0, spike1, spike2 = [[]], [[]], [[]]
batch_number, time_step = 0, 0
time_steps = 100
batch_numbers = 200
batch_size = 50
file_spike0 = open('.\\k0\\spike_num.txt')
file_spike1 = open('.\\k1\\spike_num.txt')
file_spike2 = open('.\\k2\\spike_num.txt')

for line in file_spike0.readlines():
    time_step = time_step + 1
    spike=int(line.strip('\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        spike0.append([spike])
    else:
        spike0[batch_number].append(spike)

file_spike0.close()
time_step, batch_number = 0, 0
spike0 = np.sum(np.array(spike0) / (batch_size*batch_numbers), axis=0)
spike0 = spike0[-1]

for line in file_spike1.readlines():
    time_step = time_step + 1
    spike=int(line.strip('\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        spike1.append([spike])
    else:
        spike1[batch_number].append(spike)

file_spike1.close()
time_step, batch_number = 0, 0
spike1 = np.sum(np.array(spike1) / (batch_size*batch_numbers), axis=0)
spike1 = spike1[-1]

for line in file_spike2.readlines():
    time_step = time_step + 1
    spike=int(line.strip('\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        spike2.append([spike])
    else:
        spike2[batch_number].append(spike)

file_spike2.close()
spike2 = np.sum(np.array(spike2) / (batch_size*batch_numbers), axis=0)
spike2 = spike2[-1]

##########################################################


spike_collect0, spike_collect1, spike_collect2 = [[]], [[]], [[]]
batch_number, time_step = 0, 0
time_steps = 100
batch_numbers = 200
batch_size = 50
file_spike_collect0 = open('.\\k0\\spike_collect.txt')
file_spike_collect1 = open('.\\k1\\spike_collect.txt')
file_spike_collect2 = open('.\\k2\\spike_collect.txt')

for line in file_spike_collect0.readlines():
    time_step = time_step + 1
    spike_collect=int(line.strip('.0\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        spike_collect0.append([spike_collect])
    else:
        spike_collect0[batch_number].append(spike_collect)

file_spike_collect0.close()
time_step, batch_number = 0, 0
spike_collect0 = np.sum(np.array(spike_collect0) / (batch_size*batch_numbers), axis=0)
spike_collect0 = spike_collect0[-1]

for line in file_spike_collect1.readlines():
    time_step = time_step + 1
    spike_collect=int(line.strip('.0\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        spike_collect1.append([spike_collect])
    else:
        spike_collect1[batch_number].append(spike_collect)

file_spike_collect1.close()
time_step, batch_number = 0, 0
spike_collect1 = np.sum(np.array(spike_collect1) / (batch_size*batch_numbers), axis=0)
spike_collect1 = spike_collect1[-1]

for line in file_spike_collect2.readlines():
    time_step = time_step + 1
    spike_collect=int(line.strip('.0\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        spike_collect2.append([spike_collect])
    else:
        spike_collect2[batch_number].append(spike_collect)

file_spike_collect2.close()
spike_collect2 = np.sum(np.array(spike_collect2) / (batch_size*batch_numbers), axis=0)
spike_collect2 = spike_collect2[-1]

# positive spikes
spike_pos0 = (spike0 + spike_collect0)/2
# negative spikes
spike_neg0 = (spike0 - spike_collect0)/2

spike_pos1 = (spike1 + spike_collect1)/2
spike_neg1 = (spike1 - spike_collect1)/2

spike_pos2 = (spike2 + spike_collect2)/2
spike_neg2 = (spike2 - spike_collect2)/2

# the number of spiking neurons
neuron0 = 128*24*24 + 256*24*24 + 256*12*12 + 512*12*12 + 512*6*6 + 1024*6*6 + 512*6*6 + 512*3*3 + 1024 + 512
neuron1 = neuron0
neuron2 = neuron0  
  
# auto label
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))

name_list = ['k=0','k=1','k=2','neurons']  
num_list = [spike_pos0,spike_pos1,spike_pos2,0]  
num_list1 = [spike_neg0,spike_neg1,spike_neg2,0]  
num_list2 = [0,0,0,neuron0]
x =list(range(len(num_list)))  
"""  
autolabel(plt.bar(x, num_list, label='positive spikes',fc = 'y'))
autolabel(plt.bar(x, num_list1, bottom=num_list, label='negative spikes', tick_label = name_list))
autolabel(plt.bar(x, num_list2, label='spiking neurons', fc = 'r'))
"""
plt.bar(x, num_list, label='positive spikes',fc = 'y')
plt.bar(x, num_list1, bottom=num_list, label='negative spikes', tick_label = name_list)
plt.bar(x, num_list2, label='spiking neurons', fc = 'r')

plt.legend(loc='upper left', prop=fonten)  
#text(60, -0.01, u'Di', style='italic', fontdict=fonten) 
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(" ", fontproperties=roman)
plt.ylabel("Number", fontproperties=roman) 
plt.title("Sparsity on CIFAR10", fontproperties=roman) 
  
plt.show() 

plt.savefig('scnn2_spike_neuron_cifar10.svg',format='svg')

