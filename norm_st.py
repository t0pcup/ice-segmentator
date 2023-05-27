import glob
import os
import rasterio
from tqdm import tqdm
import numpy as np
from my_lib import *
import matplotlib.pyplot as plt
from rasterio.plot import show
from PIL import Image

data_path = 'D:/'
num = '10-1'

"""
# clear_labels: mean [-14.227491, -14.227545, -27.108353, -27.108353]
#               std [5.096121, 5.0959415, 8.973816, 8.973816]

# label7...: mean [-16.2694, -16.269428, -29.840168, -29.840168]
#            std [5.957779, 5.9576864, 8.594184, 8.594184]

# label10-1: mean [-16.388807, -16.38885, -30.692194, -30.692194]
#            std [5.6070476, 5.6069245, 8.395209, 8.395208]
#            min [-49.44221, -49.44221, -49.679745, -49.679745]
#            max [16.50119, 15.677849, 2.95751, 2.9114623]
"""

# patches = glob.glob(f'{data_path}/dataset_new/label{num}/*.npy')
# xs = [[], [], [], []]
# for img in tqdm(patches):
#     im = np.load(img.replace(f'/dataset_new/label{num}', '/data')).transpose((1, 2, 0))
#     im = transforms_resize_img(image=im)['image'].transpose((2, 0, 1))
#     im = np.nan_to_num(im)
#     for channel in range(4):
#         xs[channel].append(im[channel].flatten())
#
# m, s = [], []
# mins, maxs = [], []
# for channel in range(4):
#     m.append(np.mean(xs[channel]))
#     s.append(np.std(xs[channel]))
#     mins.append(np.min(xs[channel]))
#     maxs.append(np.max(xs[channel]))
#
# print(f'{m}\n{s}')
# print(f'{mins}\n{maxs}')

patches = np.random.permutation(glob.glob(f'{data_path}/dataset_new/label{num}/*.npy'))[:50]


def save_t(full_name, im, prof):
    with rasterio.open(full_name, 'w', **prof) as src:
        src.write(im)


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


def to_img(np_arr: np.ndarray) -> Image:
    return Image.fromarray((np_arr * 255).transpose((1, 2, 0)).astype(np.uint8))


pref = 'C:/files/norm'
dir_ = os.listdir(pref)
print(dir_)
# patches = glob.glob(f'{data_path}/dataset_new/label{num}/*.npy')[:200]
arr = np.zeros(dtype=int, shape=(1280, 1280, 3))
for img in tqdm(patches):
    name = img.split(f'/label{num}\\')[1].split('.npy')[0]
    img_ = img.replace(f'/dataset_new/label{num}', '/data')

    x = np.load(img)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # print(i, j, str(x[i, j]))
            arr[i, j] = palette0[str(x[i, j])]
    mask = Image.fromarray(np.uint8(arr))

    # m_stand_m_norm = new_normalize(np.load(img_))
    # m_stand_s_norm = new_normalize(np.load(img_), single_norm=True)
    s_stand_m_norm = new_normalize(np.load(img_))
    s_stand_p_norm = new_normalize(np.load(img_), plural=True)
    s_stand_s_norm = new_normalize(np.load(img_), single_norm=True)

    # m_stand_m_norm = stand(m_stand_m_norm)
    # m_stand_s_norm = stand(m_stand_s_norm)
    """s_stand_m_norm = stand(s_stand_m_norm, single_stand=True)
    s_stand_p_norm = stand(s_stand_p_norm, single_stand=True)
    s_stand_s_norm = stand(s_stand_s_norm, single_stand=True)

    fig = plt.figure(dpi=250)
    ax1 = fig.add_subplot(2, 2, 1)
    plt.imshow(to_img(s_stand_m_norm))
    to_img(s_stand_m_norm).show(title='multi')
    ax2 = fig.add_subplot(2, 2, 2)
    plt.imshow(to_img(s_stand_p_norm))
    to_img(s_stand_p_norm).show(title='plural')
    ax3 = fig.add_subplot(2, 2, 3)
    plt.imshow(to_img(s_stand_s_norm))
    to_img(s_stand_s_norm).show(title='single')
    ax4 = fig.add_subplot(2, 2, 4)
    plt.imshow(mask)
    plt.show()"""

    sat_img = rasterio.open(img_.replace('.npy', '.tiff'), 'r')
    profile = sat_img.profile
    profile['count'] = 4
    # save_t(f'{pref}/{dir_[1]}/{name}.tiff', m_stand_m_norm * 255, profile)
    # save_t(f'{pref}/{dir_[2]}/{name}.tiff', m_stand_s_norm * 255, profile)
    """save_t(f'{pref}/{dir_[3]}/{name}.tiff', s_stand_m_norm * 255, profile)
    save_t(f'{pref}/{dir_[4]}/{name}.tiff', s_stand_p_norm * 255, profile)
    save_t(f'{pref}/{dir_[5]}/{name}.tiff', s_stand_s_norm * 255, profile)
    mask.save(f"{pref}/{dir_[0]}/{name}.gif")"""

    save_t(f'{pref}/no_stand/m_norm/{name}.tiff', s_stand_m_norm * 255, profile)
    save_t(f'{pref}/no_stand/p_norm/{name}.tiff', s_stand_p_norm * 255, profile)
    save_t(f'{pref}/no_stand/s_norm/{name}.tiff', s_stand_s_norm * 255, profile)
    mask.save(f"{pref}/no_stand/label/{name}.gif")
