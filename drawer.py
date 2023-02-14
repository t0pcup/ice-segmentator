import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import time

# Writer will output to ./runs/ directory by default
writer = SummaryWriter('E:/files/history')
phase = 'test'

for epo in range(100):
    writer.add_scalar(phase + ' | loss', np.random.random(), epo)
    writer.add_scalar(phase + ' | f1', np.random.random(), epo)
    writer.add_scalar(phase + ' | b_accuracy', np.random.random(), epo)
    writer.add_scalar(phase + ' | accuracy', np.random.random(), epo)
    writer.add_scalar(phase + ' | mean_IoU', np.random.random(), epo)
    time.sleep(0.1)
    epo += 1

writer.close()
