import glob
import rasterio
from tqdm import tqdm
import numpy as np
from my_lib import *

data_path = 'D:/'

# for img_file in tqdm(glob.glob(f'{data_path}/data/*EA*.tiff')[:10]):
#     try:
#         sat_img = rasterio.open(img_file, 'r')
#         profile = sat_img.profile
#         profile['count'] = 3
#         np_full_name = img_file.replace('.tiff', '.npy')
#         npy_name = np_full_name.split('\\')[1]
#
#         im_dst = np.load(np_full_name.replace('data', 'dataset/label'))
#         msk = palette0[im_dst[:][:]].astype(np.uint8)
#         im_dst = transforms_resize_lbl(image=im_dst, mask=msk)['mask'].transpose((2, 0, 1))
#         save_tiff(f'D:/view/map/{npy_name.replace(".npy", ".tiff")}', im_dst, profile)
#
#         images = np.empty(shape=(4, 1280, 1280))
#         images[:] = (normalize(np.load(np_full_name)) * 255).astype(np.uint8)
#         im_dst = np.asarray(images)[:3]
#         im_dst = im_dst.transpose((1, 2, 0))
#         im_dst = transforms_resize_img(image=im_dst)['image']
#         im_dst = im_dst.transpose((2, 0, 1))
#         save_tiff(f'D:/view/image/{npy_name.replace(".npy", ".tiff")}', im_dst, profile)
#     except:
#         continue

import matplotlib.pyplot as plt
from rasterio.plot import show
from PIL import Image


def new_normalize(im_: np.ndarray, single_norm=False, plural=False) -> np.ndarray:
    im_ = np.nan_to_num(im_)
    mean = np.array([-16.388807, -16.38885, -30.692194, -30.692194])
    std = np.array([5.6070476, 5.6069245, 8.395209, 8.395208])
    if plural:
        mean = np.array([-14.227491, -14.227545, -27.108353, -27.108353])
        std = np.array([5.096121, 5.0959415, 8.973816, 8.973816])

    if single_norm:
        mean, std = np.zeros(im_.shape[0]), np.zeros(im_.shape[0])
        for channel in range(im_.shape[0]):
            mean[channel] = np.mean(im_[channel, :, :])
            std[channel] = np.std(im_[channel, :, :])

    norm = torchvision.transforms.Normalize(mean, std)
    return np.asarray(norm.forward(torch.from_numpy(im_)))


def stand(im_: np.ndarray, single_stand=False) -> np.ndarray:
    im_ = np.nan_to_num(im_)
    min_ = np.array([-49.44221, -49.44221, -49.679745, -49.679745])
    max_ = np.array([16.50119, 15.677849, 2.95751, 2.9114623])
    if single_stand:
        min_, max_ = np.zeros(im_.shape[0]), np.zeros(im_.shape[0])
        for channel in range(im_.shape[0]):
            min_[channel] = np.min(im_[channel, :, :])
            max_[channel] = np.max(im_[channel, :, :])

    for channel in range(im_.shape[0]):
        im_[channel] = (im_[channel] - min_[channel]) / (max_[channel] - min_[channel])
    return im_


num = '10-5'
patches = np.random.permutation(glob.glob(f'{data_path}/dataset_new/label{num}/*.npy'))  # [:20]
pal = {
    '0': [0, 0, 0],  # ignore
    '1': [0, 0, 255],  # водичка и low ct
    '5': [255, 255, 0],  # med ct
    '2': [255, 128, 0],  # ct >= 70%
    '3': [255, 0, 0],  # fast ice
    '4': [32, 128, 0],  # land
    # '5': [255, 255, 255],

    # '3': [128, 255, 128],
    # '4': [255, 255, 0],
    # '5': [255, 0, 0],
    # '6': [255, 255, 255],
}
# pal = {
#     '0': [0, 0, 0],  # none
#     '1': [0, 255, 0],  # land
#     '2': [255, 0, 0],  # water
#     '3': [255, 255, 255],  # shelf
#
#     '4': [25, 33, 177],  # ice free
#     '5': [77, 84, 216],  # open water
#     '6': [114, 119, 216],  # bergy water
#
#     '7': [15, 79, 168],  # 1
#     '8': [67, 128, 211],
#     '9': [105, 150, 211],
#
#     '10': [0, 162, 135],  # 2
#     '11': [52, 208, 182],
#     '12': [94, 208, 189],
#
#     '13': [174, 241, 0],  # 3
#     '14': [196, 248, 62],
#     '15': [210, 248, 112],
#
#     '16': [255, 236, 0],  # 4
#     '17': [255, 241, 64],
#     '18': [255, 245, 115],
#
#     '19': [255, 205, 0],  # 5
#     '20': [255, 217, 64],
#     '21': [255, 227, 115],
#
#     '22': [255, 170, 0],  # 6
#     '23': [255, 191, 64],
#     '24': [255, 208, 115],
#
#     '25': [255, 137, 0],  # 7
#     '26': [255, 166, 64],
#     '27': [255, 190, 115],
#
#     '28': [255, 76, 0],  # 8
#     '29': [255, 121, 64],
#     '30': [255, 157, 115],
#
#     '31': [225, 0, 76],  # 9
#     '32': [240, 60, 121],  # 9-10
#     '33': [240, 108, 152],  # 10
# }
ban = []
d = dict((str(1 + k), 0) for k in [-1, 0, 1, 2, 3, 4])
arr = np.zeros(dtype=int, shape=(256, 256, 3))
"""special = '_ignore=-1'"""
special = ''
got_em = glob.glob(f'{data_path}/dataset_new/label{num}/*.npy'.replace(f'/label{num}', f'/label{num}{special}_0'))
# p = [n for n in patches if n.replace(f'/label{num}', f'/label{num}{special}_0') not in got_em]
full = 256 * 256
# for lbl in tqdm(patches):
#     l_ = np.load(lbl)  # - 1  # TODO sub 1 for stats ignore index
#     x = transforms_resize_lbl(image=l_, mask=l_)['mask']
#     name = lbl.split(f'/label{num}\\')[1].split('.npy')[0]
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             arr[i, j] = pal[str(x[i, j])]
#
#     a = np.unique(x, return_counts=True)
#     for i in range(len(a[0])):
#         try:
#             if str(a[0][i]) in ['4'] and a[1][i] >= full * 0.9:  # and 3 not in np.unique(x)
#                 ban.append(lbl)
#                 continue
#             d[str(a[0][i])] += a[1][i]
#         except:
#             continue
#     if lbl not in ban:
#         x[x == 0] = -1
#         x[x == 1] = 0
#         x[x == 5] = 1
#         # Image.fromarray(np.uint8(arr)).save(f"D:/dataset_new/label{num}{special}_1/{name}.gif")
#         np.save(lbl.replace(f'/label{num}', f'/label{num}{special}_0'), x)
#
# print(d)
# plt.pie(d.values(), labels=d.keys())
# plt.axis('equal')
# plt.title(f'{num} | all regions, amt of patches: {len(patches)}')
# plt.show()
#
# del d['0']
# plt.pie(list(d.values()), labels=list(d.keys()))
# plt.axis('equal')
# plt.title(f'{num} | drop ignore_index')
# plt.show()
# print(ban)

im_patches = glob.glob(f'{data_path}/dataset_new/label{num}_1/*.gif')
for img in tqdm(im_patches):
    # im = new_normalize(np.load(img.replace(f'/dataset_new/label{num}', '/data')), plural=True).transpose((1, 2, 0))
    # im = transforms_resize_img(image=im)['image'].transpose(2, 0, 1)
    # im = stand(im, single_stand=True)
    im = np.load(img.replace(f'/dataset_new/label{num}_1', '/data').replace('.gif', '.npy')).transpose((1, 2, 0))
    im = transforms_resize_img(image=im)['image'].transpose(2, 0, 1)
    im = new_normalize(im, plural=True)
    im = stand(im, single_stand=True)
    np.save(img.replace(f'/label{num}_1', f'/data{num}').replace('.gif', '.npy'), im)

    r = np.load(img.replace(f'/label{num}_1', f'/data{num}').replace('.gif', '.npy'))[:3]
    r = (r * 255).transpose((1, 2, 0))
    im = Image.fromarray(r.astype(np.uint8))
    name = img.split(f'/label{num}_1\\')[1].split('.gif')[0]
    im.save(f"D:/dataset_new/data{num}_1/{name}.gif")
    # im.show()
