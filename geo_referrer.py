import os
import rasterio.warp
import numpy as np
import warnings
from PIL import Image

warnings.filterwarnings("ignore")


def save_tiff(full_name, im):
    global profile
    with rasterio.open(full_name, 'w', **profile) as src:
        src.write(im)


path = 'E:/files/'
min_v, max_v = -84.64701, 4.379311
classes = ['land', 'water', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'no data', 'ice shelf', 'undefined']
palette0 = np.array([[50, 200, 50],    # land
                     [50, 50, 255],    # water
                     [0, 0, 0],        # 2
                     [28, 28, 28],     # 3
                     [57, 57, 57],     # 4
                     [85, 85, 85],     # 5
                     [114, 114, 114],  # 6
                     [142, 142, 142],  # 7
                     [171, 171, 171],  # 8
                     [199, 199, 199],  # 9
                     [228, 228, 228],  # 10
                     [255, 255, 0],    # no data
                     [255, 255, 255],  # ice shelf
                     [50, 50, 50]])    # undefined

names = os.listdir(f'{path}view/image/')

for file in os.listdir(f'{path}view/raw_pred/'):  # f'{path}data/'  # [:40]
    if 'npy' in file[:-4]:
        continue
    name = file.split('.')[0]

    profile = rasterio.open(f'{path}data/{name}.tiff', 'r').profile
    profile['count'] = 3

    # images = np.empty(shape=(4, 1280, 1280))
    # images[:] = ((np.load(f'{path}data/{name}.npy') + min_v) / (max_v - abs(min_v) + 1) * 255).astype(np.uint8)
    # im_dst = np.asarray(images)[:3]
    # im_dst = im_dst.transpose((1, 2, 0))
    # tr = albumentations.Compose([albumentations.Resize(640, 640, interpolation=3)])(image=im_dst)
    # im_dst = tr['image']
    # im_dst = im_dst.transpose((2, 0, 1))
    # save_tiff(f'{path}view/image/{name}.tiff', im_dst)  # , mask=label[:, :]

    # try:
    #     im_dst = np.load(f'{path}label/{name}.npy')
    #     im_dst = palette0[im_dst[:][:]].astype(np.uint8).transpose((2, 0, 1))
    #     save_tiff(f'{path}view/map/{name}.tiff', im_dst)
    # except Exception:
    #     print(1)

    try:
        im_dst = np.asarray(Image.open(f'{path}view/raw_pred/{name}.gif'))
        im_dst = palette0[im_dst[:][:]].astype(np.uint8).transpose((2, 0, 1))
        save_tiff(f'{path}view/predict/{name}.tiff', im_dst)
    except Exception:
        print(1)
