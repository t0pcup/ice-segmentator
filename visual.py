# print('Formula of grading: 0.25 * (КР1 * 0.6 + ДЗ1 * 0.4) + 0.75 * (КР2 * 0.2 + ДЗ2 * 0.2 + ЭКЗ * 0.6)')
# kr_1, kr_2 = 9, 9
# hw_1, hw_2 = 9, 0
# exam = 0
#
# print(f'Current grade:\t\t{0.25 * (kr_1*0.6 + hw_1*0.4) + 0.75 * (kr_2*0.2 + hw_2*0.2 + exam*0.6)}')
# print(f'Aiming:\t\t\t\t{0.25 * (kr_1*0.6 + hw_1*0.4) + 0.75 * (kr_2*0.2 + 8*0.2 + 6*0.6)}')
# print(f'Best possible:\t\t{0.25 * (kr_1*0.6 + hw_1*0.4) + 0.75 * (kr_2*0.2 + 10*0.2 + 10*0.6)}')
# print(f'Worst acceptable:\t{0.25 * (kr_1*0.6 + hw_1*0.4) + 0.75 * (kr_2*0.2 + 7*0.2 + 2*0.6)}')

# path = r'E:/dafuck/'
# img = np.load(path+'image/2021-07-30_4991.npy')  # 2021-06-12_5851
# lbl = np.load(path+'label/2021-07-30_4991.npy')

# print(np.unique(img[0]), img[0].min(), img[0].max(), img[0].mean())
# print(np.unique(img[1]), img[1].min(), img[1].max(), img[1].mean())
# print(np.unique(img[2]), img[2].min(), img[2].max(), img[2].mean())
# print(np.unique(img[3]), img[3].min(), img[3].max(), img[3].mean())
# print(img.shape)

# vc = dict()
# for l in listdir(path+'label/'):
#     keys = np.unique(np.load(f'{path}label/{l}'))
#     values = np.unique(np.load(f'{path}label/{l}'), return_counts=True)[1]
#     for i in range(len(keys)):
#         if keys[i] not in vc.keys():
#             vc[keys[i]] = 0
#         vc[keys[i]] += values[i]
#
# print(vc)
# print(sum(list(vc.values()))/256/256, len(listdir(path+'label/')))

# print(np.unique(lbl, return_counts=True), lbl.min(), lbl.max(), lbl.mean())
# print(lbl.shape)

import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations
import segmentation_models_pytorch as smp

transforms_val = [albumentations.Resize(1280, 1280)]  # 64, 64 512, 512


class InferDataset(Dataset):
    def __init__(self, datapath_, file_n):
        self.datapath = datapath_
        self.data_list = [file_n]  # os.listdir(f'{datapath_}/label0')
        self.transforms_val = albumentations.Compose(transforms_val)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = np.load(f'{self.datapath}/data/{self.data_list[index]}')
        label = np.load(f'{self.datapath}/label/{self.data_list[index]}')

        # img = np.load(f'{self.datapath}/data/{self.data_list[index]}')
        # label = np.load(f'{self.datapath}/label0/{self.data_list[index]}')
        # min_v, max_v = np.min(img), np.max(img)
        min_v, max_v = -84.64701, 4.379311
        img = ((img + min_v) / (abs(max_v) - abs(min_v) + 1) * 255).astype(np.uint8)

        img = img.transpose(1, 2, 0)
        augmented = self.transforms_val(image=img, mask=label[:, :])
        img = augmented['image']
        label = augmented['mask']
        img = img.transpose(2, 0, 1)
        return img, label


def collate_fn(batch):
    imgs_, labels_ = [], []
    for _, sample in enumerate(batch):
        img, label = sample
        imgs_.append(torch.from_numpy(img.copy()))
        labels_.append(torch.from_numpy(label.copy()))
    return torch.stack(imgs_, 0).type(torch.FloatTensor), torch.stack(labels_, 0).type(torch.LongTensor)


def unnormalize(image):
    np_img = image.numpy()
    min_v, max_v = -84.64701, 4.379311
    np_img = ((np_img + min_v) / (abs(max_v) - abs(min_v) + 1) * 255).astype(np.uint8)
    np_img = np.transpose(np_img, (1, 2, 0))
    np_img = (np_img * 255).astype(np.uint8)
    return np_img


datapath = r'E:/files'
palette = np.array([i for i in range(14)])
print(palette)
model_path = r'E:/files/pts/NewLbl_v0.1.pth'
# model_path = r'E:/dafuck/pts/DeepLabV3Plus_50.pth'
BATCH_SIZE = 1
device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = smp.DeepLabV3(
    encoder_name="timm-mobilenetv3_small_minimal_100",  # efficientnet-b0
    encoder_weights=None,
    in_channels=4,
    classes=14,
)

for file_name in os.listdir(datapath + '/label')[:30]:
    print(file_name)

    model.to(device)
    state_dict = torch.load(model_path)['model_state_dict']
    model.load_state_dict(state_dict, strict=False)

    dataset = InferDataset(datapath, file_name)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)  # 30.01.2023 was True

    inputs, labels = next(iter(dataloader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    model.eval()
    ce = nn.CrossEntropyLoss()
    outputs = model(inputs)
    y_pred = torch.argmax(outputs, dim=1)

    name = file_name.split('.')[0]
    im = Image.fromarray(palette[y_pred[0, :, :]])
    print(np.unique(y_pred[0, :, :], return_counts=True))
    # im.show()
    im.save(f"E:/files/view/raw_pred/{name}.gif")

    # im = Image.fromarray(palette[labels[0, :, :]] * 255)
    # # im.show()
    # im.save(f"E:/dafuck/view/visualize/{name}l.gif")
    # im = Image.fromarray(unnormalize(inputs[0, :, :, :]) * 255)
    # # im.show()
    # im.save(f"E:/dafuck/view/visualize/{name}i.gif")

# inputs, labels = next(iter(dataloader))
#
# inputs = inputs.to(device)
# labels = labels.to(device)
#
# model.eval()
# ce = nn.CrossEntropyLoss()
# outputs = model(inputs)
# y_pred = torch.argmax(outputs, dim=1)
#
# palette0 = np.array([[0,   0,   0],   # 0 no info
#                     [0,   0, 255],   # 1 bl
#                     [0, 255,   0],   # 2 gr
#                     [255, 255,   0],   # 3 ye
#                     [255, 130,   0],   # 4 or
#                     [255,   0,   0],   # 5 re
#                     [255, 100, 130],   # 6 pi
#                     [255,   0, 255]])  # 7 pu
#
# palette = np.array([0, 0.5, 1])
#
# # Отрисуем первый батч
# fig, axs = plt.subplots(4, BATCH_SIZE, figsize=(BATCH_SIZE * 2.8, 8))
# o = 0.1
# for i in range(BATCH_SIZE):
#     im = Image.fromarray(palette[y_pred[i, :, :]]*255)
#     # im.show()
#     im.save(f"E:/dafuck/view/visualize/{o}p.gif")
#     im = Image.fromarray(palette[labels[i, :, :]]*255)
#     # im.show()
#     im.save(f"E:/dafuck/view/visualize/{o}l.gif")
#     # im.save(f"E:/dafuck/view/map/{i}.gif")
#     im = Image.fromarray(unnormalize(inputs[i, :, :, :])*255)
#     # im.show()
#     im.save(f"E:/dafuck/view/visualize/{o}i.gif")
#     # im.save(f"E:/dafuck/view/image/{i}.gif")
#
#     # axs[0, i].imshow(palette[y_pred[i, :, :]])
#     # axs[1, i].imshow(palette[labels[i, :, :]])
#     # axs[2, i].imshow(palette[y_pred[i, :, :]], 'gray', interpolation='none', alpha=0.5)
#     # axs[2, i].imshow(palette[labels[i, :, :]], 'jet', interpolation='none', alpha=0.7)
#     # axs[3, i].imshow(unnormalize(inputs[i, :, :, :]), 'jet', interpolation='none')
#
# # fig.suptitle('Сравнение истинных значений масок с предсказанными', fontsize=16)
# # plt.show()
