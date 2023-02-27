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

reg = ''
d = dict((k, 0) for k in range(8))
patches = glob.glob(f'{data_path}/dataset_new/label4/*{reg}*.npy')
for lbl in tqdm(patches):
    a = np.unique(np.load(lbl), return_counts=True)
    for i in range(len(a[0])):
        try:
            d[int(a[0][i])] += a[1][i]
        except:
            continue

print(d)

plt_label = list(map(str, d.keys()))
plt.pie(list(d.values()), labels=plt_label)
plt.axis('equal')
plt.title(f'all regions, amt of patches: {len(patches)}')
plt.show()
