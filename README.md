# Structural exploration of spiking neural networks under direct training
## Abstract
This paper delves into the structural optimization of Spiking Neural Networks (SNNs) by 
innovating neuron models and optimizing training methods, with a particular focus on 
performance enhancement under direct training frameworks.
Firstly, the paper introduces the PMLF and 3PMLF structures based on the PLIF model and 
MLF model. These structures enhance the specificity of network units by adjusting the internal 
states of neurons. Additionally, to address issues of inaccurate gradient propagation and 
premature convergence during network training, the paper introduces a threshold-dependent 
Spiking Residual Network structure and improves the training stability and performance through 
threshold-dependent Batch Normalization (tdBN) methods.
In the training methodology section, the paper employs the Time-Efficient Training (TET) 
method. This method optimizes the loss function, adjusts the weight distribution of time steps, 
and reduces error accumulation and extreme outliers during the training process.
Experimental validation demonstrates the effectiveness of the proposed model on various 
datasets: achieving a 0.64% performance improvement on the static dataset CIFAR-10, and 
3.12% and 3.3% performance improvements on the dynamic datasets DVS-Gesture and DVSCIFAR10, respectively.
Through this research, the paper not only validates the effectiveness of the new structures 
and training methods but also provides practical guidance and theoretical foundations for the 
future application of Spiking Neural Networks in handling complex cognitive tasks. These 
achievements showcase how meticulous structural design and method optimization can 
effectively enhance the performance of SNNs in real-time information processing and temporal 
tasks.

## 1. Datasets
* Download link of DVS-gesture: https://research.ibm.com/interactive/dvsgesture/.
Put file `DvsGesture.tar.gz` in the path `./data/DVS_Gesture/source_DvsGesture/`, then unzip `DvsGesture.tar.gz` (`tar -xzvf DvsGesture.tar.gz`).

* Download link of CIFAR10-DVS: https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671/2.
Put file `airplane.zip`～`horse.zip` in the path `./data/DVS_CIFAR10/source_DvsCIFAR10/`, then unzip `airplane.zip`～`horse.zip`.

* CIFAR10 dataset can be downloaded online.

## 2. Dependencies:
* python 3.7.10
* numpy 1.19.5
* torch 1.9.0+cu111
* torchvision 0.10.0+cu111
* tensorboardX 2.4
* h5py 3.3.0

## 3. Preprocessing
DVS-gesture and CIFAR10-DVS need to be pre-processed. The syntax is as follow,
```
python DVS_CIFAR10_preprocess.py
python DVS_Gesture_preprocess.py
```

## 4. Traning
To train a new model, the basic syntax is like:
```
python train_for_cifar10.py
python train_for_gesture.py
python train_for_dvscifar10.py
```
