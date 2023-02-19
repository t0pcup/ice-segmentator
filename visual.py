import os
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from my_lib import *


class InferDataset(Dataset):
    def __init__(self, path, file_n):
        self.path = path
        self.data_list = [file_n]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        f_name = self.data_list[index]
        image, label = item_getter(self.path, f_name)
        return image, label


def un_normalize(np_img: np.ndarray):
    np_img = np.transpose(np_img, (1, 2, 0))
    np_img = (np_img * 255).astype(np.uint8)
    return np_img.transpose(2, 0, 1)


data_path = r'E:/files'
palette = np.array([i for i in range(8)])
model_path = r'E:/files/pts/versions/XXX.pth'
BATCH_SIZE = 1

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

    # im = Image.fromarray(un_normalize(inputs[0, :, :, :]) * 255)
    # im.save(f"E:/files/view/raw_img/{name}i.gif")
