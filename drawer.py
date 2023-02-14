# import torch
# import torchvision
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import datasets, transforms
# import numpy as np
# import time
# import os
#
# # Writer will output to ./runs/ directory by default
# dir_ = 'E:/files/history'
# run_name = '2'
# os.makedirs(f"{dir_}/{run_name}", exist_ok=True)
# writer = SummaryWriter(f"{dir_}/{run_name}")
# phase = 'train'
#
# for epo in range(150):
#     writer.add_scalar(phase + ' | loss', np.random.random(), epo)
#     writer.add_scalar(phase + ' | f1', np.random.random(), epo)
#     writer.add_scalar(phase + ' | b_accuracy', np.random.random(), epo)
#     writer.add_scalar(phase + ' | accuracy', np.random.random(), epo)
#     writer.add_scalar(phase + ' | mean_IoU', np.random.random(), epo)
#     time.sleep(1)
#     epo += 1
#
# writer.close()
