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

patches = ['D://dataset_new/label10-1\\HB_0607T230517_49_5_5.npy',
           'D://dataset_new/label10-1\\EA_0301T232157_22_7_2.npy',
           'D://dataset_new/label10-1\\HB_0222T214919_20_4_3.npy',
           'D://dataset_new/label10-1\\EA_0906T225835_23_3_7.npy',
           'D://dataset_new/label10-1\\EA_1122T230547_29_2_5.npy',
           'D://dataset_new/label10-1\\HB_1122T230522_28_5_4.npy',
           'D://dataset_new/label10-1\\EC_1220T214058_58_10_5.npy',
           'D://dataset_new/label10-1\\EA_0524T232159_45_2_3.npy',
           'D://dataset_new/label10-1\\EC_0315T221151_10_3_3.npy',
           'D://dataset_new/label10-1\\EA_0315T230538_32_5_3.npy',
           'D://dataset_new/label10-1\\EC_1227T213226_6_3_3.npy',
           'D://dataset_new/label10-1\\HB_0412T214116_21_7_4.npy',
           'D://dataset_new/label10-1\\EA_0830T230546_55_6_1.npy',
           'D://dataset_new/label10-1\\HB_0412T214116_21_1_3.npy',
           'D://dataset_new/label10-1\\EA_0607T230542_50_1_5.npy',
           'D://dataset_new/label10-1\\EA_0301T232312_25_9_2.npy',
           'D://dataset_new/label10-1\\EA_0607T230607_51_1_5.npy',
           'D://dataset_new/label10-1\\EC_0315T221151_10_1_2.npy',
           'D://dataset_new/label10-1\\EA_1122T230547_29_4_5.npy',
           'D://dataset_new/label10-1\\HB_0315T230538_32_4_6.npy',
           'D://dataset_new/label10-1\\EA_0906T225835_23_4_4.npy',
           'D://dataset_new/label10-1\\HB_0517T214922_44_7_5.npy',
           'D://dataset_new/label10-1\\HB_0111T214925_6_8_5.npy',
           'D://dataset_new/label10-1\\EC_1213T214907_49_8_4.npy',
           'D://dataset_new/label10-1\\EA_0607T230607_51_7_7.npy',
           'D://dataset_new/label10-1\\EC_0426T212432_16_4_6.npy',
           'D://dataset_new/label10-1\\HB_0517T214922_44_5_1.npy',
           'D://dataset_new/label10-1\\EC_0419T213220_25_5_6.npy',
           'D://dataset_new/label10-1\\EA_1108T232256_26_10_5.npy',
           'D://dataset_new/label10-1\\EA_0315T230538_32_8_3.npy',
           'D://dataset_new/label10-1\\EC_0222T214850_19_4_5.npy',
           'D://dataset_new/label10-1\\EA_1108T232321_27_5_6.npy',
           'D://dataset_new/label10-1\\EC_0111T214925_6_9_4.npy',
           'D://dataset_new/label10-1\\EC_0419T213245_26_7_6.npy',
           'D://dataset_new/label10-1\\EC_0222T214850_19_9_6.npy',
           'D://dataset_new/label10-1\\HB_1122T230522_28_6_2.npy',
           'D://dataset_new/label10-1\\EC_0426T212317_13_7_3.npy',
           'D://dataset_new/label10-1\\EA_0614T225802_52_8_1.npy',
           'D://dataset_new/label10-1\\HB_1122T230547_29_7_5.npy',
           'D://dataset_new/label10-1\\GL_0201T230006_17_5_3.npy',
           'D://dataset_new/label10-1\\EA_1122T230547_29_7_2.npy',
           'D://dataset_new/label10-1\\HB_0607T230517_49_6_2.npy',
           'D://dataset_new/label10-1\\EC_0419T213155_24_6_6.npy',
           'D://dataset_new/label10-1\\HB_0315T230513_31_7_5.npy',
           'D://dataset_new/label10-1\\HB_1122T230522_28_7_3.npy',
           'D://dataset_new/label10-1\\EC_0426T212317_13_6_1.npy',
           'D://dataset_new/label10-1\\EA_0830T230546_55_8_2.npy',
           'D://dataset_new/label10-1\\HB_0517T214853_43_8_2.npy',
           'D://dataset_new/label10-1\\GL_0118T231655_14_9_4.npy',
           'D://dataset_new/label10-1\\EC_0517T214922_44_7_5.npy',
           'D://dataset_new/label10-1\\EA_0301T232157_22_7_3.npy',
           'D://dataset_new/label10-1\\EC_1227T213226_6_9_2.npy',
           'D://dataset_new/label10-1\\EC_0111T214925_6_7_1.npy',
           'D://dataset_new/label10-1\\EA_0301T232312_25_4_6.npy',
           'D://dataset_new/label10-1\\EC_0308T222030_53_5_5.npy',
           'D://dataset_new/label10-1\\EA_0524T232314_48_3_3.npy',
           'D://dataset_new/label10-1\\EC_0412T214001_18_6_3.npy',
           'D://dataset_new/label10-1\\EA_0830T230546_55_3_2.npy',
           'D://dataset_new/label10-1\\EC_0104T215620_0_6_6.npy',
           'D://dataset_new/label10-1\\EC_0426T212407_15_8_3.npy',
           'D://dataset_new/label10-1\\EA_1108T232256_26_3_3.npy',
           'D://dataset_new/label10-1\\EC_0419T213245_26_6_4.npy',
           'D://dataset_new/label10-1\\EC_0222T223616_28_6_1.npy',
           'D://dataset_new/label10-1\\HB_0315T230513_31_6_2.npy',
           'D://dataset_new/label10-1\\EC_0412T213936_17_9_1.npy',
           'D://dataset_new/label10-1\\EA_0301T232222_23_10_5.npy',
           'D://dataset_new/label10-1\\EA_0830T230546_55_5_2.npy',
           'D://dataset_new/label10-1\\HB_0830T230546_55_8_5.npy',
           'D://dataset_new/label10-1\\EC_0419T213220_25_8_3.npy',
           'D://dataset_new/label10-1\\EA_0301T232157_22_8_2.npy',
           'D://dataset_new/label10-1\\EA_0301T232157_22_6_3.npy',
           'D://dataset_new/label10-1\\HB_0412T214116_21_5_2.npy',
           'D://dataset_new/label10-1\\EA_0614T225831_53_4_7.npy',
           'D://dataset_new/label10-1\\EA_0906T225835_23_6_2.npy',
           'D://dataset_new/label10-1\\EA_1108T232256_26_8_5.npy',
           'D://dataset_new/label10-1\\EA_0315T230538_32_4_5.npy',
           'D://dataset_new/label10-1\\EC_0419T213155_24_6_4.npy',
           'D://dataset_new/label10-1\\EC_0419T213220_25_3_3.npy',
           'D://dataset_new/label10-1\\HB_0412T214116_21_7_2.npy',
           'D://dataset_new/label10-1\\EC_0315T221151_10_5_3.npy',
           'D://dataset_new/label10-1\\EA_1108T232321_27_5_5.npy',
           'D://dataset_new/label10-1\\EC_0426T212317_13_4_6.npy',
           'D://dataset_new/label10-1\\EA_0524T232314_48_5_4.npy',
           'D://dataset_new/label10-1\\EA_0614T225831_53_8_4.npy',
           'D://dataset_new/label10-1\\HB_0607T230517_49_7_1.npy',
           'D://dataset_new/label10-1\\EC_0419T213155_24_5_5.npy',
           'D://dataset_new/label10-1\\HB_0222T214850_19_5_3.npy',
           'D://dataset_new/label10-1\\EA_0524T232314_48_6_5.npy',
           'D://dataset_new/label10-1\\EC_0419T213130_23_2_3.npy',
           'D://dataset_new/label10-1\\EC_0111T214810_3_6_5.npy',
           'D://dataset_new/label10-1\\EC_0419T213155_24_4_4.npy',
           'D://dataset_new/label10-1\\HB_0412T214116_21_8_2.npy',
           'D://dataset_new/label10-1\\EC_0322T220353_10_4_6.npy',
           'D://dataset_new/label10-1\\GL_0111T232445_8_3_2.npy',
           'D://dataset_new/label10-1\\EC_0419T213155_24_6_2.npy',
           'D://dataset_new/label10-1\\EC_0419T213245_26_9_3.npy',
           'D://dataset_new/label10-1\\GL_0201T230031_18_8_9.npy',
           'D://dataset_new/label10-1\\EA_0315T230513_31_6_1.npy',
           'D://dataset_new/label10-1\\EC_0222T214850_19_6_1.npy',
           'D://dataset_new/label10-1\\HB_0111T214925_6_7_2.npy']
# patches = np.random.permutation(glob.glob(f'{data_path}/dataset_new/label{num}/*.npy'))[:5]


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
