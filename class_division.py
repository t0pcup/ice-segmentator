import warnings
from shapely.geometry import box
import glob
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.plot import show
import numpy as np
import fiona
import os
from my_lib import normalize, transforms_resize_img, save_tiff, palette0

warnings.filterwarnings("ignore")
reg_path = 'C:/files/regions/2021'
# view_path = 'E:/files/view'
data_path = 'D:/data'
translate_classes_simple = {
    '55': 1,  # ice free
    '01': 8,  # open water
    '02': 9,  # bergy water

    '10': 2,  # 1/10
    '12': 2,  # 1-2/10
    '13': 2,  # 1-3/10
    '20': 2,  # 2/10
    '23': 2,  # 2-3/10
    '24': 2,  # 2-4/10
    '30': 2,  # 3/10
    '34': 2,  # 3-4/10
    '35': 2,  # 3-5/10
    '40': 2,  # 4/10
    '45': 2,  # 4-5/10
    '46': 2,  # 4-6/10

    '50': 3,  # 5/10
    '56': 3,  # 5-6/10
    '57': 3,  # 5-7/10
    '60': 3,  # 6/10
    '67': 3,  # 6-7/10
    '68': 3,  # 6-8/10
    '70': 3,  # 7/10
    '78': 3,  # 7-8/10
    '79': 3,  # 7-9/10
    '80': 3,  # 8/10
    '89': 3,  # 8-9/10
    '81': 3,  # 8-10/10

    '90': 4,  # 9/10
    '91': 5,  # 9-10/10
    '92': 6,  # 10/10
}


def def_num(it: dict) -> int:
    # undefined / land / no data => zero
    trans_dict = {'L': 0, 'W': 1, 'N': 0, 'S': 7}
    try:
        return trans_dict[it['POLY_TYPE']]
    except:
        return translate_classes_simple[it['CT']]


ban_list = []
for img_file in tqdm(glob.glob(f'{data_path}/*.tiff')):
    code = img_file.split('\\')[1].split('T')[0]
    reg_name, date = code.split('_')
    bad_img_flag = False
    for f in glob.glob(f'{reg_path}/*{reg_name}*2021{date}*.shp'):
        shapes_list = gpd.read_file(f).to_crs('epsg:4326')
        file = fiona.open(f)
        sat_img = rasterio.open(img_file, 'r')
        bb, profile = sat_img.bounds, sat_img.profile
        sat_poly = box(*bb, ccw=True)

        geom_value = []
        for i in range(len(file)):
            geom_ = shapes_list.geometry[i].intersection(sat_poly)
            if geom_.area == 0:
                continue
            prop = file[i]
            geom_value.append((geom_, def_num(prop['properties'])))

        if len(geom_value) == 0:
            bad_img_flag = True
            continue

        rasterized = features.rasterize(geom_value, out_shape=sat_img.shape, transform=sat_img.transform,
                                        all_touched=True, fill=0, merge_alg=MergeAlg.replace, dtype=np.int16)

        # if rasterized.sum() == 0:
        #     bad_img_flag = True
        #     continue
        if (rasterized == 0).all():  # or (rasterized == 1).all() or (rasterized == 6).all()
            bad_img_flag = True
            continue

        if len(np.unique(rasterized)) > 4:
            fig, ax = plt.subplots(1, figsize=(10, 10))
            show(rasterized, ax=ax)
            plt.gca().invert_yaxis()
            plt.show()

        np_full_name = img_file.replace('.tiff', '.npy')
        npy_name = np_full_name.split('\\')[1]
        np.save(f'D:/dataset_new/label4/{npy_name}', rasterized)

        # profile = sat_img.profile
        # profile['count'] = 3
        #
        # images = np.empty(shape=(4, 1280, 1280))
        # images[:] = (normalize(np.load(np_full_name)) * 255).astype(np.uint8)
        # im_dst = np.asarray(images)[:3]
        # im_dst = im_dst.transpose((1, 2, 0))
        # im_dst = transforms_resize_img(image=im_dst)['image']
        # im_dst = im_dst.transpose((2, 0, 1))
        # save_tiff(f'E:/files/view/image/{npy_name.replace(".npy", "")}.tiff', im_dst, profile)
        #
        # try:
        #     im_dst = rasterized
        #     im_dst = palette0[im_dst[:][:]].astype(np.uint8).transpose((2, 0, 1))
        #     save_tiff(f'E:/files/view/map/{npy_name.replace(".npy", "")}.tiff', im_dst, profile)
        # except:
        #     continue
    if bad_img_flag:
        ban_list.append(img_file)

# for file in ban_list:
#     os.remove(file)
#     os.remove(file.replace('.tiff', '.npy'))
