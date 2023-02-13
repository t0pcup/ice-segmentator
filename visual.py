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

# from main import coll_fn, normalize

transforms_val = [
    albumentations.Resize(1000, 1000, p=0.5, interpolation=3),
    albumentations.RandomBrightnessContrast(contrast_limit=0.1, brightness_by_max=False),
    albumentations.RandomRotate90(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.RandomCrop(640, 640, always_apply=False, p=1.0)
]


def normalize(im_: np.ndarray) -> np.ndarray:
    # print(np.unique(im)) TODO nan in raw images
    im_ = np.nan_to_num(im_)
    mean, std = np.zeros(im_.shape[0]), np.zeros(im_.shape[0])
    for channel in range(im_.shape[0]):
        mean[channel] = np.mean(im_[channel, :, :])
        std[channel] = np.std(im_[channel, :, :])

    norm = torchvision.transforms.Normalize(mean, std)
    return np.asarray(norm.forward(torch.from_numpy(im_)))


def coll_fn(batch_):
    ims_, labels_ = [], []
    for _, sample in enumerate(batch_):
        im, lbl = sample
        ims_.append(torch.from_numpy(im.copy()))
        labels_.append(torch.from_numpy(lbl.copy()))
    return torch.stack(ims_, 0).type(torch.FloatTensor), torch.stack(labels_, 0).type(torch.LongTensor)


class InferDataset(Dataset):
    def __init__(self, datapath_, file_n):
        self.datapath = datapath_
        self.data_list = [file_n]
        self.transforms_val = albumentations.Compose(transforms_val)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        f_name = self.data_list[index]

        img = normalize(np.load(f'{self.datapath}/data/{f_name}'))
        # img = np.nan_to_num(img) TODO
        label = np.load(f'{self.datapath}/label/{f_name}')

        img = img.transpose((1, 2, 0))
        augmented = self.transforms_val(image=img, mask=label[:, :])
        img = augmented['image']
        label = augmented['mask']

        img = img.transpose(2, 0, 1)
        return img, label


def unnormalize(np_img: np.ndarray):
    np_img = np.transpose(np_img, (1, 2, 0))
    np_img = (np_img * 255).astype(np.uint8)
    return np_img.transpose(2, 0, 1)


data_path = r'E:/files'
palette = np.array([i for i in range(8)])
model_path = r'E:/files/pts/versions/iou_zapal_na_1_f1_grthan_acc.pth'
BATCH_SIZE = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = smp.DeepLabV3(
    encoder_name="timm-mobilenetv3_small_minimal_100",  # efficientnet-b0
    encoder_weights=None,
    in_channels=4,
    classes=8,
)

for file_name in os.listdir(data_path + '/label')[:40]:
    print(file_name)

    model.to(device)
    state_dict = torch.load(model_path)['model_state_dict']
    model.load_state_dict(state_dict, strict=False)

    dataset = InferDataset(data_path, file_name)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=coll_fn, shuffle=False)  # 30.01.2023 was True

    inputs, labels = next(iter(dataloader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    model.eval()
    ce = nn.CrossEntropyLoss()
    outputs = model(inputs)
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
