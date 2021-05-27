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

accuracies0, accuracies1, accuracies2 = [[]], [[]], [[]]
batch_number, time_step = 0, 0
time_steps = 40
batch_numbers = 200
batch_size = 50
file_accuracy0 = open('.\\k0\\accuracy.txt')
file_accuracy1 = open('.\\k1\\accuracy.txt')
file_accuracy2 = open('.\\k2\\accuracy.txt')

for line in file_accuracy0.readlines():
    time_step = time_step + 1
    accuracy=int(line.strip('\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        accuracies0.append([accuracy])
    else:
        accuracies0[batch_number].append(accuracy)

file_accuracy0.close()
time_step, batch_number = 0, 0
accuracies0 = np.sum(np.array(accuracies0), axis=0)/(batch_size*batch_numbers)

for line in file_accuracy1.readlines():
    time_step = time_step + 1
    accuracy=int(line.strip('\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        accuracies1.append([accuracy])
    else:
        accuracies1[batch_number].append(accuracy)

file_accuracy1.close()
time_step, batch_number = 0, 0
accuracies1 = np.sum(np.array(accuracies1), axis=0)/(batch_size*batch_numbers)

for line in file_accuracy2.readlines():
    time_step = time_step + 1
    accuracy=int(line.strip('\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        accuracies2.append([accuracy])
    else:
        accuracies2[batch_number].append(accuracy)

file_accuracy2.close()
accuracies2 = np.sum(np.array(accuracies2), axis=0)/(batch_size*batch_numbers)

# classification accuracy of full-precision ANN, need to be updated
accuracies3 = []
for i in range(len(accuracies0)):
    accuracies3.append(0.9793)

names = range(len(accuracies0))
#x = range(len(names))
x0 = range(len(accuracies0))
x1 = range(len(accuracies1))
x2 = range(len(accuracies2))
x3 = range(len(accuracies3))

#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
plt.xlim(0, time_steps) 
plt.ylim(0, 1.0)  
plt.plot(x0, accuracies0, marker='o', ms=8,label='k=0: accuracy=97.80%, time steps=8')
plt.plot(x1, accuracies1, marker='*', ms=8,label='k=1: accuracy=97.86%, time steps=14')
plt.plot(x2, accuracies2, marker='^', ms=8,label='k=2: accuracy=97.51%, time steps=13')
plt.plot(x3, accuracies3, marker='_', ms=8,label='Full-precision ANN: accuracy=97.93%') # need to be updated
plt.legend(loc='lower right', prop=fonten)  
#text(60, -0.01, u'Di', style='italic', fontdict=fonten) 
plt.xticks(x0[::2], names[::2], rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("Time step", fontproperties=roman) 
plt.ylabel("Accuracy", fontproperties=roman) 
plt.title("Accuracy versus Speed on N-MNIST", fontproperties=roman) 

plt.show()

plt.savefig('scnn_accuracy_n_mnist.svg',format='svg')
