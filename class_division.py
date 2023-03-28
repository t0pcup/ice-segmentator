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
translate_classes = {
    '55': 4,  # ice free
    '01': 5,  # open water
    '02': 6,  # bergy water

    '10': 7,  # 1/10
    '12': 8,  # 1-2/10
    '13': 9,  # 1-3/10

    '20': 10,  # 2/10
    '23': 11,  # 2-3/10
    '24': 12,  # 2-4/10

    '30': 13,  # 3/10
    '34': 14,  # 3-4/10
    '35': 15,  # 3-5/10

    '40': 16,  # 4/10
    '45': 17,  # 4-5/10
    '46': 18,  # 4-6/10

    '50': 19,  # 5/10
    '56': 20,  # 5-6/10
    '57': 21,  # 5-7/10

    '60': 22,  # 6/10
    '67': 23,  # 6-7/10
    '68': 24,  # 6-8/10

    '70': 25,  # 7/10
    '78': 26,  # 7-8/10
    '79': 27,  # 7-9/10

    '80': 28,  # 8/10
    '89': 29,  # 8-9/10
    '81': 30,  # 8-10/10

    '90': 31,  # 9/10
    '91': 32,  # 9-10/10
    '92': 33,  # 10/10
}

translate_classes_simple = {
    '55': 0,  # ice free
    '01': 0,  # open water

    '02': 1,  # bergy water | FA 10 | icebergs

    '10': 0,  # 1/10 | noname FA
    '12': 0,  # 1-2/10
    '13': 0,  # 1-3/10

    '20': 2,  # 2/10 | FA 03-05
    '23': 2,  # 2-3/10
    '24': 2,  # 2-4/10
    '30': 2,  # 3/10 | FA 03-05
    '34': 2,  # 3-4/10
    '35': 2,  # 3-5/10
    '40': 2,  # 4/10 | ???
    '45': 2,  # 4-5/10
    '46': 2,  # 4-6/10
    # ++++++++++++++
    '50': 2,  # 5/10 | FA 03-05
    '56': 2,  # 5-6/10
    '57': 2,  # 5-7/10
    '60': 2,  # 6/10 | ???
    '67': 2,  # 6-7/10
    '68': 2,  # 6-8/10

    '70': 3,  # 7/10 | FA 03-07
    '78': 3,  # 7-8/10
    '79': 3,  # 7-9/10
    '80': 3,  # 8/10 | FA 03-05
    '89': 3,  # 8-9/10
    '81': 3,  # 8-10/10
    '90': 3,  # 9/10 | FA 03-06

    '91': 4,  # 9-10/10 | FA 03-08 TODO
    '92': 4,  # 10/10 | FA 03-08 TODO
}

codes = []
dic = dict(zip(translate_classes_simple.keys(), [[] for _ in translate_classes_simple.keys()]))


def def_num(it: dict) -> int:
    # undefined / land / no data => zero
    trans_dict = {'L': 4, 'W': 1, 'N': 0, 'S': 3}  # no shelves in src
    try:
        ct, fa = int(it['CT']), int(it['FA'])
        CT_FA_stat.append(it['CT'] + '_' + it['FA'])
    except:
        _ = 0
    try:
        return trans_dict[it['POLY_TYPE']]
    except:
        if it['FA'] == '08':
            return 3
        if it['CT'] in ['55', '01', '02']:
            return 1
        if it['FA'] == '10':
            return 1
        if it['CT'] in ['10', '20', '30', '12', '13', '23', '24', '34', '35', '40', '45', '46',
                        '50', '56', '57', '60', '67', '68', '70', '78', '79']:
            return 5
        if it['CT'] in ['80', '89', '81', '90', '91', '92']:
            return 2

        print(it['FA'], it['CT'])
        # return 2
        print(it['POLY_TYPE'], it['CT'], it['FA'])


CT_FA_stat = []
l_ = list(np.random.permutation(glob.glob(f'{data_path}/*.tiff')))[::-1]
for img_file in tqdm(l_):
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
        # if (rasterized == 0).all() or (rasterized == 1).all():  # or (rasterized == 6).all()
        d = dict((k, 0) for k in range(6))
        a = np.unique(rasterized, return_counts=True)
        assert not np.any(np.isnan(a))
        for i in range(len(a[0])):
            try:
                d[int(a[0][i])] += a[1][i]
            except:
                continue

        full = 1280 * 1280
        if d[0] + d[4] >= full or d[0] + d[1] >= full:
            bad_img_flag = True
            continue

        if len(np.unique(rasterized)) > 3:
            fig, ax = plt.subplots(1, figsize=(10, 10))
            show(rasterized, ax=ax)
            plt.gca().invert_yaxis()
            plt.show()

        np_full_name = img_file.replace('.tiff', '.npy')
        npy_name = np_full_name.split('\\')[1]
        np.save(f'D:/dataset_new/label10-5/{npy_name}', rasterized)

        # print(*codes)
        codes = []

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

print(np.unique(np.asarray(CT_FA_stat), return_counts=True))
print(dic)
