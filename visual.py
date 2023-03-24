import os
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from my_lib import *
from segmentation_models_pytorch.metrics import get_stats, iou_score, accuracy, balanced_accuracy, f1_score


class InferDataset(Dataset):
    def __init__(self, path, file_n):
        self.path = path
        self.data_list = [file_n]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        f_name = self.data_list[index]
        image, label = item_getter(self.path, f_name, val=False)
        return image, label


def un_normalize(np_img: np.ndarray):
    np_img = np.transpose(np_img, (1, 2, 0))
    np_img = (np_img * 255).astype(np.uint8)
    return np_img.transpose(2, 0, 1)


# palette = np.array([i for i in range(8)])
data_path = 'D:/dataset_new'
model_path = 'D:/pts_24.03/v10-2+_ign_Unet_resnet18_v10-2+_ign_Unet_resnet18_2.pth'
mod = 'resnet'
BATCH_SIZE = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_t(full_name, im, prof):
    with rasterio.open(full_name, 'w', **prof) as src:
        src.write(im)


lst = list(np.random.permutation(os.listdir(data_path + '/label10-2_ignore=-1_0')))
lst = ['EA_0301T232157_22_1_3.npy']
for file_name in tqdm(lst[:1]):
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

    name = file_name.split('.')[0]

    sat_img = rasterio.open(f'D:/data/{name}.tiff', 'r')
    profile = sat_img.profile
    profile['count'] = 4
    images = np.empty(shape=(4, 256, 256))
    images[:] = np.asarray(inputs[0, :, :])
    save_t(f"D:/M/10-2+/i/{name}.tiff", np.asarray(images) * 255, profile)  # image

    profile['count'] = 3
    save_t(f"D:/M/10-2+/p_{mod}_real/{name}.tiff", palette0[1 + y_pred[0, :, :]].astype(np.uint8).transpose((2, 0, 1)),
           profile)  # predict
    # y_pred[0, labels[0, :, :] == -1] = -1
    # save_t(f"D:/M/10-2+/p_{mod}/{name}.tiff", palette0[1 + y_pred[0, :, :]].astype(np.uint8).transpose((2, 0, 1)),
    #        profile)  # predict

    save_t(f"D:/M/10-2+/l_{mod}/{name}.tiff", palette0[1 + labels[0, :, :]].astype(np.uint8).transpose((2, 0, 1)),
           profile)  # label

    output = torch.LongTensor(y_pred[0, :, :])
    target = torch.LongTensor(labels[0, :, :])
    try:
        print(sum(sum(np.asarray(torch.LongTensor(y_pred[0, :, :] == -1)))))
    except:
        print(np.unique(y_pred[0, :, :]))
        pass
    print(sum(sum(np.asarray(torch.LongTensor(labels[0, :, :] == -1)))))
    st = get_stats(output, target, mode='multiclass', num_classes=5)
    print(name)
    for class_ in range(5):
        stats = []
        if np.sum(np.asarray(st[2])[:, class_]) == 0:
            continue
        for s in range(len(st)):  # TP  FP  FN  TN
            arr = np.asarray(st[s])
            print(['TP', 'FP', 'FN', 'TN'][s], end='\t')
            print(np.sum(arr[:, class_]))
            stats.append((st[s])[:, class_])
            # stats.append(np.asarray([arr[:, class_]]))
            # print(arr[:, 0])

        # print(tuple(stats)[0].shape)
        print(float(f1_score(*tuple(stats), reduction='macro') * inputs.size(0)))
        print(float(f1_score(*st, reduction='macro') * inputs.size(0)))

    # im = Image.fromarray(palette0[1 + y_pred[0, :, :]].astype(np.uint8))
    # im.save(f"D:/M/10-2+/p/{name}.gif")

    # im = Image.fromarray(palette0[1 + labels[0, :, :]].astype(np.uint8))
    # im.save(f"D:/M/10-2+/l/{name}.gif")

    # im_dst = im_dst.transpose((1, 2, 0))
    # im_dst = transforms_resize_img(image=im_dst)['image']
    # im_dst = im_dst.transpose((2, 0, 1))

    # im = Image.fromarray(im_dst.astype(np.uint8))
    # im.save(f"D:/M/10-2+/i/{name}.gif")

    # arr = np.load(f"D:/data/{name}.npy")[:3].transpose((1, 2, 0))
    # im = Image.fromarray(arr.astype(np.uint8))
    # im.save(f"D:/view/image/{name}.gif")

    # im = Image.fromarray(un_normalize(inputs[0, :, :, :]) * 255)
    # im.save(f"E:/files/view/raw_img/{name}i.gif")
