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
from main import coll_fn, normalize

transforms_val = [albumentations.Resize(1280, 1280)]


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


def unnormalize(image):
    np_img = normalize(image.numpy())
    np_img = np.transpose(np_img, (1, 2, 0))
    np_img = (np_img * 255).astype(np.uint8)
    return np_img.transpose(2, 0, 1)


data_path = r'E:/files'
palette = np.array([i for i in range(14)])
print(palette)
model_path = r'E:/files/pts/NewLbl_v0.1.pth'
BATCH_SIZE = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = smp.DeepLabV3(
    encoder_name="timm-mobilenetv3_small_minimal_100",  # efficientnet-b0
    encoder_weights=None,
    in_channels=4,
    classes=14,
)

for file_name in os.listdir(data_path + '/label')[:30]:
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
