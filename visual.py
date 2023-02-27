import os
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from my_lib import *


class InferDataset(Dataset):
    def __init__(self, path, file_n):
        self.path = path
        self.data_list = [file_n]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        f_name = self.data_list[index]
        image, label = item_getter(self.path, f_name, val=True)
        return image, label


def un_normalize(np_img: np.ndarray):
    np_img = np.transpose(np_img, (1, 2, 0))
    np_img = (np_img * 255).astype(np.uint8)
    return np_img.transpose(2, 0, 1)


palette = np.array([i for i in range(8)])
data_path = 'D:/dataset'
model_path = 'D:/NewLbl_17.pth'
BATCH_SIZE = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for file_name in tqdm(os.listdir(data_path + '/val_label')):
    model_ft.to(device)
    state_dict = torch.load(model_path, map_location=device)['model_state_dict']
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
    labels = np.asarray(labels.cpu())

    if len(np.unique(labels)) < 3:
        continue

    name = file_name.split('.')[0]
    im = Image.fromarray(palette0[y_pred[0, :, :]].astype(np.uint8))
    im.save(f"D:/M/p/{name}.gif")

    im = Image.fromarray(palette0[labels[0, :, :]].astype(np.uint8))
    im.save(f"D:/M/l/{name}.gif")

    images = np.empty(shape=(4, 128, 128))
    images[:] = normalize(inputs[0, :, :]) * 255
    im_dst = np.asarray(images)[:3]
    im_dst = im_dst.transpose((1, 2, 0))
    im_dst = transforms_resize_img(image=im_dst)['image']
    # im_dst = im_dst.transpose((2, 0, 1))
    # print(im_dst.shape)
    # print(im_dst.astype(np.uint8))
    # print(type(im_dst))
    im = Image.fromarray(im_dst.astype(np.uint8))
    im.save(f"D:/M/i/{name}.gif")

    # im = Image.fromarray(un_normalize(inputs[0, :, :, :]) * 255)
    # im.save(f"E:/files/view/raw_img/{name}i.gif")
