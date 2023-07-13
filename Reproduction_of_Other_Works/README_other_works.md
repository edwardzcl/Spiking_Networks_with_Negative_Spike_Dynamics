# Reproduction_of_other_works

***
**This code can be used as the supplemental material for the paper: "Towards a Lossless Conversion for Spiking Neural Networks with Negative Spike Dynamics". (Submitted to *Advanced Intelligent Systems, Wiley*, July, 2023)** .
***

## Citation:
To be completed.

### **Features**:
- In many other works, they use various input encoding styles (e.g constant and possion codes) and network training parameters which are different from ours, even there are no available open-source projects to reproduce their experimental results. Besides, in their works, `Signal Noise Ratio` (SNR) for spiking activities of SNNs was rarely discussed. Hence, we choose as many as possible reproducible works for comparison.


## File overview:
- `README.md` - this readme file.<br>

`Other folders` include six kinds of different reproducible projects for SNR (input noise on datasets) experiments in our paper, please refer to the `noise_ratio` character in their respective main function files.<br>

## Requirements
### **Dependencies and Libraries**:
* python 3.8 (https://www.python.org/ or https://www.anaconda.com/)
* tensorflow_gpu 1.2.1 (https://github.com/tensorflow)
* tensorlayer 1.8.5 (https://github.com/tensorlayer)
* pytorch 1.7.1 (https://pytorch.org/)
* matlab R2017b (https://www.mathworks.com/)
* CPU: Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
* GPU: Tesla V100

**Please refer to their readme files for more information.**

### **Installation**:
To install requirements, **please refer to their readme files for more information.**
### **Datasets**:
* MNIST: [dataset](http://yann.lecun.com/exdb/mnist/), [preprocessing](https://github.com/tensorlayer/tensorlayer/blob/1.8.5/tensorlayer/files.py)
* FashionMNIST: [dataset](https://github.com/zalandoresearch/fashion-mnist), 
[preprocessing](https://github.com/tensorlayer/tensorlayer/blob/1.8.5/tensorlayer/files.py)
* CIFAR10: [dataset](https://www.cs.toronto.edu/~kriz/), 
[preprocessing](https://github.com/tensorlayer/tensorlayer/blob/1.8.5/tensorlayer/files.py)

### **Run the code**:
**Please refer to their readme files for more information.**
Besides, you can refer to the `noise_ratio` character in their respective main function files.

## Online available open-source projects
Our provided six projects in this repository can be directly reproducible, and you can easily adjust the `noise_ratio` variable in their main function files for SNR experiments.
Besides, you can refer to their original papers for more available information.

## Results
Please check these results in our paper.



## More question:<br>
- There might be a little difference of results for multiple training repetitions, because of the randomization. 
- Please feel free to reach out here or email: xxx@xxx, if you have any questions or difficulties. I'm happy to help guide you.
