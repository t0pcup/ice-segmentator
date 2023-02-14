import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import albumentations
import segmentation_models_pytorch as smp
from my_lib import coll_fn, item_getter, model_ft

transforms_ = [
    albumentations.CenterCrop(640, 640, always_apply=False, p=1.0),
    # albumentations.RandomRotate90(p=0.5),
    # albumentations.HorizontalFlip(p=0.5),
]


class InferDataset(Dataset):
    def __init__(self, datapath_, file_n):
        self.datapath = datapath_
        self.data_list = [file_n]
        self.transforms_val = albumentations.Compose(transforms_)
        # self.transforms_img = albumentations.Compose(transforms_resize_img)
        # self.transforms_lbl = albumentations.Compose(transforms_resize_lbl)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        f_name = self.data_list[index]
        image, label = item_getter(self.datapath, f_name, self.transforms_val)
        # img = normalize(np.nan_to_num(np.load(f'{self.datapath}/data/{f_name}')))
        # label = np.load(f'{self.datapath}/label/{f_name}')
        #
        # img = img.transpose((1, 2, 0))
        # augmented = self.transforms_val(image=img, mask=label[:, :])
        # # img, label = augmented['image'], augmented['mask']
        # img = self.transforms_img(image=augmented['image'], mask=augmented['mask'])['image']
        # label = self.transforms_lbl(image=augmented['image'], mask=augmented['mask'])['mask']
        # img = img.transpose(2, 0, 1)
        return image, label


def unnormalize(np_img: np.ndarray):
    np_img = np.transpose(np_img, (1, 2, 0))
    np_img = (np_img * 255).astype(np.uint8)
    return np_img.transpose(2, 0, 1)


data_path = r'E:/files'
palette = np.array([i for i in range(8)])
model_path = r'E:/files/pts/versions/iou_zapal_na_1_f1_grthan_acc.pth'
BATCH_SIZE = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_ft = smp.DeepLabV3(
#     encoder_name="timm-mobilenetv3_small_minimal_100",  # efficientnet-b0
#     encoder_weights=None,
#     in_channels=4,
#     classes=8,
# )

for file_name in os.listdir(data_path + '/label')[:40]:
    print(file_name)

    model_ft.to(device)
    state_dict = torch.load(model_path)['model_state_dict']
    model_ft.load_state_dict(state_dict, strict=False)

    dataset = InferDataset(data_path, file_name)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=coll_fn, shuffle=False)  # 30.01.2023 was True

    inputs, labels = next(iter(dataloader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    model_ft.eval()
    ce = nn.CrossEntropyLoss()
    outputs = model_ft(inputs)
    y_pred = torch.argmax(outputs, dim=1).cpu()

    name = file_name.split('.')[0]
    im = Image.fromarray(palette[y_pred[0, :, :]])
    # im.show()
    im.save(f"E:/files/view/raw_pred/{name}.gif")

    labels = np.asarray(labels.cpu())
    im = Image.fromarray(palette[labels[0, :, :]])
    im.save(f"E:/files/view/raw_map/{name}.gif")

    # im = Image.fromarray(unnormalize(inputs[0, :, :, :]) * 255)
    # im.save(f"E:/files/view/raw_img/{name}i.gif")
