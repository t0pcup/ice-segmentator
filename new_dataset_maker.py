import warnings
from shapely.ops import cascaded_union, unary_union
from shapely.geometry import Polygon, box
import os
import glob
from eodag.api.core import EODataAccessGateway
from eodag import setup_logging
import geopandas as gpd
import tqdm
from eoreader.reader import Reader
from eoreader.bands import *
import rasterio
import matplotlib.pyplot as plt
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.plot import show
import numpy as np
import fiona

warnings.filterwarnings("ignore")
reg_path = 'E:/files/regions/2021'
view_path = 'E:/files/view'
dataset_path = 'E:/files/dataset'

setup_logging(1)
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__USERNAME"] = "katarina.spasenovic@omni-energy.it"
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__PASSWORD"] = "M@rkon!1997"
dag = EODataAccessGateway()
workspace = os.path.join(view_path, "/quicklooks")
if not os.path.isdir(workspace):
    os.mkdir(workspace)

yaml_content = """
peps:
    download:
        outputs_prefix: "{}"
        extract: true
""".format(workspace)

with open(os.path.join(workspace, 'eodag_conf.yml'), "w") as f_yml:
    f_yml.write(yaml_content.strip())

dag = EODataAccessGateway(os.path.join(workspace, 'eodag_conf.yml'))
product_type = 'S1_SAR_GRD'
dag.set_preferred_provider("peps")
test_name = 'E:/dafuck/regions/2021/cis_SGRDREA_20210614T1800Z_pl_a.shp'
cnt = 0
# white_list_shapes, white_list_images = [], []
white_list_shapes = ['E:/files/regions/2021/cis_SGRDRWA_20210614T1800Z_pl_a.shp',
                     'E:/files/regions/2021/cis_SGRDRWA_20210628T1800Z_pl_a.shp',
                     'E:/files/regions/2021/cis_SGRDRWA_20210712T1800Z_pl_a.shp',
                     'E:/files/regions/2021/cis_SGRDRWA_20210719T1800Z_pl_a.shp',
                     'E:/files/regions/2021/cis_SGRDRWA_20210802T1800Z_pl_a.shp',
                     'E:/files/regions/2021/cis_SGRDRWA_20210809T1800Z_pl_a.shp',
                     'E:/files/regions/2021/cis_SGRDRWA_20210823T1800Z_pl_a.shp',
                     'E:/files/regions/2021/cis_SGRDRWA_20210830T1800Z_pl_a.shp',
                     'E:/files/regions/2021/cis_SGRDRWA_20210920T1800Z_pl_a.shp',
                     'E:/files/regions/2021/cis_SGRDRWA_20210927T1800Z_pl_a.shp',
                     'E:/files/regions/2021/cis_SGRDRWA_20211011T1800Z_pl_a.shp',
                     'E:/files/regions/2021/cis_SGRDRWA_20211018T1800Z_pl_a.shp']

white_list_images = ['E:/files/view/image/2021-06-14_0_2.tiff', 'E:/files/view/image/2021-06-14_0_3.tiff',
                     'E:/files/view/image/2021-06-14_0_4.tiff', 'E:/files/view/image/2021-06-14_0_5.tiff',
                     'E:/files/view/image/2021-06-14_0_6.tiff', 'E:/files/view/image/2021-06-14_0_7.tiff',
                     'E:/files/view/image/2021-06-14_0_8.tiff', 'E:/files/view/image/2021-06-14_1_2.tiff',
                     'E:/files/view/image/2021-06-14_1_3.tiff', 'E:/files/view/image/2021-06-14_1_4.tiff',
                     'E:/files/view/image/2021-06-14_1_6.tiff', 'E:/files/view/image/2021-06-14_1_7.tiff',
                     'E:/files/view/image/2021-06-14_2_2.tiff', 'E:/files/view/image/2021-06-14_2_3.tiff',
                     'E:/files/view/image/2021-06-14_2_4.tiff', 'E:/files/view/image/2021-06-14_3_1.tiff',
                     'E:/files/view/image/2021-06-14_3_2.tiff', 'E:/files/view/image/2021-06-14_3_3.tiff',
                     'E:/files/view/image/2021-06-14_3_4.tiff', 'E:/files/view/image/2021-06-14_3_5.tiff',
                     'E:/files/view/image/2021-06-14_4_1.tiff', 'E:/files/view/image/2021-06-14_4_2.tiff',
                     'E:/files/view/image/2021-06-14_4_4.tiff', 'E:/files/view/image/2021-06-14_5_2.tiff',
                     'E:/files/view/image/2021-06-28_1_4.tiff', 'E:/files/view/image/2021-06-28_2_3.tiff',
                     'E:/files/view/image/2021-06-28_2_4.tiff', 'E:/files/view/image/2021-06-28_2_5.tiff',
                     'E:/files/view/image/2021-06-28_2_6.tiff', 'E:/files/view/image/2021-06-28_3_3.tiff',
                     'E:/files/view/image/2021-06-28_3_4.tiff', 'E:/files/view/image/2021-06-28_3_5.tiff',
                     'E:/files/view/image/2021-06-28_3_6.tiff', 'E:/files/view/image/2021-06-28_4_3.tiff',
                     'E:/files/view/image/2021-06-28_4_4.tiff', 'E:/files/view/image/2021-06-28_5_2.tiff',
                     'E:/files/view/image/2021-06-28_5_3.tiff', 'E:/files/view/image/2021-06-28_5_4.tiff',
                     'E:/files/view/image/2021-06-28_6_2.tiff', 'E:/files/view/image/2021-06-28_6_3.tiff',
                     'E:/files/view/image/2021-07-12_7_2.tiff', 'E:/files/view/image/2021-07-12_7_3.tiff',
                     'E:/files/view/image/2021-07-12_7_4.tiff', 'E:/files/view/image/2021-07-12_7_5.tiff',
                     'E:/files/view/image/2021-07-12_8_2.tiff', 'E:/files/view/image/2021-07-12_8_3.tiff',
                     'E:/files/view/image/2021-07-12_8_4.tiff', 'E:/files/view/image/2021-07-12_8_5.tiff',
                     'E:/files/view/image/2021-07-19_10_3.tiff', 'E:/files/view/image/2021-07-19_2_2.tiff',
                     'E:/files/view/image/2021-07-19_2_3.tiff', 'E:/files/view/image/2021-07-19_2_4.tiff',
                     'E:/files/view/image/2021-07-19_3_2.tiff', 'E:/files/view/image/2021-07-19_3_3.tiff',
                     'E:/files/view/image/2021-07-19_3_4.tiff', 'E:/files/view/image/2021-07-19_4_2.tiff',
                     'E:/files/view/image/2021-07-19_4_3.tiff', 'E:/files/view/image/2021-07-19_4_4.tiff',
                     'E:/files/view/image/2021-07-19_5_2.tiff', 'E:/files/view/image/2021-07-19_5_3.tiff',
                     'E:/files/view/image/2021-07-19_5_4.tiff', 'E:/files/view/image/2021-07-19_6_2.tiff',
                     'E:/files/view/image/2021-07-19_6_3.tiff', 'E:/files/view/image/2021-07-19_6_4.tiff',
                     'E:/files/view/image/2021-07-19_7_2.tiff', 'E:/files/view/image/2021-07-19_7_3.tiff',
                     'E:/files/view/image/2021-07-19_7_4.tiff', 'E:/files/view/image/2021-07-19_7_5.tiff',
                     'E:/files/view/image/2021-07-19_8_2.tiff', 'E:/files/view/image/2021-07-19_8_3.tiff',
                     'E:/files/view/image/2021-07-19_8_4.tiff', 'E:/files/view/image/2021-07-19_8_5.tiff',
                     'E:/files/view/image/2021-07-19_9_3.tiff', 'E:/files/view/image/2021-07-19_9_4.tiff',
                     'E:/files/view/image/2021-07-19_9_5.tiff', 'E:/files/view/image/2021-08-02_1_1.tiff',
                     'E:/files/view/image/2021-08-02_2_1.tiff', 'E:/files/view/image/2021-08-02_2_2.tiff',
                     'E:/files/view/image/2021-08-02_2_3.tiff', 'E:/files/view/image/2021-08-02_2_4.tiff',
                     'E:/files/view/image/2021-08-02_3_2.tiff', 'E:/files/view/image/2021-08-02_3_3.tiff',
                     'E:/files/view/image/2021-08-02_3_4.tiff', 'E:/files/view/image/2021-08-02_4_2.tiff',
                     'E:/files/view/image/2021-08-02_4_3.tiff', 'E:/files/view/image/2021-08-02_4_4.tiff',
                     'E:/files/view/image/2021-08-02_5_2.tiff', 'E:/files/view/image/2021-08-02_5_3.tiff',
                     'E:/files/view/image/2021-08-02_5_4.tiff', 'E:/files/view/image/2021-08-02_6_3.tiff',
                     'E:/files/view/image/2021-08-02_6_4.tiff', 'E:/files/view/image/2021-08-02_6_5.tiff',
                     'E:/files/view/image/2021-08-02_7_3.tiff', 'E:/files/view/image/2021-08-02_7_4.tiff',
                     'E:/files/view/image/2021-08-02_7_5.tiff', 'E:/files/view/image/2021-08-02_8_2.tiff',
                     'E:/files/view/image/2021-08-02_8_3.tiff', 'E:/files/view/image/2021-08-02_8_4.tiff',
                     'E:/files/view/image/2021-08-02_9_3.tiff', 'E:/files/view/image/2021-08-02_9_4.tiff',
                     'E:/files/view/image/2021-08-09_4_2.tiff', 'E:/files/view/image/2021-08-09_5_2.tiff',
                     'E:/files/view/image/2021-08-09_6_2.tiff', 'E:/files/view/image/2021-08-09_7_1.tiff',
                     'E:/files/view/image/2021-08-09_7_2.tiff', 'E:/files/view/image/2021-08-09_8_1.tiff',
                     'E:/files/view/image/2021-08-09_8_2.tiff', 'E:/files/view/image/2021-08-09_8_3.tiff',
                     'E:/files/view/image/2021-08-09_9_1.tiff', 'E:/files/view/image/2021-08-09_9_2.tiff',
                     'E:/files/view/image/2021-08-09_9_3.tiff', 'E:/files/view/image/2021-08-23_10_4.tiff',
                     'E:/files/view/image/2021-08-23_1_3.tiff', 'E:/files/view/image/2021-08-23_2_3.tiff',
                     'E:/files/view/image/2021-08-23_3_3.tiff', 'E:/files/view/image/2021-08-23_4_2.tiff',
                     'E:/files/view/image/2021-08-23_4_3.tiff', 'E:/files/view/image/2021-08-23_4_4.tiff',
                     'E:/files/view/image/2021-08-23_5_2.tiff', 'E:/files/view/image/2021-08-23_5_3.tiff',
                     'E:/files/view/image/2021-08-23_5_4.tiff', 'E:/files/view/image/2021-08-23_6_2.tiff',
                     'E:/files/view/image/2021-08-23_6_3.tiff', 'E:/files/view/image/2021-08-23_6_4.tiff',
                     'E:/files/view/image/2021-08-23_7_1.tiff', 'E:/files/view/image/2021-08-23_7_2.tiff',
                     'E:/files/view/image/2021-08-23_7_3.tiff', 'E:/files/view/image/2021-08-23_7_4.tiff',
                     'E:/files/view/image/2021-08-23_8_1.tiff', 'E:/files/view/image/2021-08-23_8_2.tiff',
                     'E:/files/view/image/2021-08-23_8_3.tiff', 'E:/files/view/image/2021-08-23_8_4.tiff',
                     'E:/files/view/image/2021-08-23_9_1.tiff', 'E:/files/view/image/2021-08-23_9_2.tiff',
                     'E:/files/view/image/2021-08-23_9_3.tiff', 'E:/files/view/image/2021-08-23_9_4.tiff',
                     'E:/files/view/image/2021-08-30_10_5.tiff', 'E:/files/view/image/2021-08-30_10_6.tiff',
                     'E:/files/view/image/2021-08-30_1_4.tiff', 'E:/files/view/image/2021-08-30_2_3.tiff',
                     'E:/files/view/image/2021-08-30_2_4.tiff', 'E:/files/view/image/2021-08-30_2_5.tiff',
                     'E:/files/view/image/2021-08-30_3_3.tiff', 'E:/files/view/image/2021-08-30_3_4.tiff',
                     'E:/files/view/image/2021-08-30_3_5.tiff', 'E:/files/view/image/2021-08-30_3_6.tiff',
                     'E:/files/view/image/2021-08-30_4_2.tiff', 'E:/files/view/image/2021-08-30_4_3.tiff',
                     'E:/files/view/image/2021-08-30_4_4.tiff', 'E:/files/view/image/2021-08-30_4_5.tiff',
                     'E:/files/view/image/2021-08-30_4_6.tiff', 'E:/files/view/image/2021-08-30_5_2.tiff',
                     'E:/files/view/image/2021-08-30_5_3.tiff', 'E:/files/view/image/2021-08-30_5_4.tiff',
                     'E:/files/view/image/2021-08-30_5_5.tiff', 'E:/files/view/image/2021-08-30_5_6.tiff',
                     'E:/files/view/image/2021-08-30_6_2.tiff', 'E:/files/view/image/2021-08-30_6_3.tiff',
                     'E:/files/view/image/2021-08-30_6_4.tiff', 'E:/files/view/image/2021-08-30_6_5.tiff',
                     'E:/files/view/image/2021-08-30_7_1.tiff', 'E:/files/view/image/2021-08-30_7_2.tiff',
                     'E:/files/view/image/2021-08-30_7_3.tiff', 'E:/files/view/image/2021-08-30_7_4.tiff',
                     'E:/files/view/image/2021-08-30_7_5.tiff', 'E:/files/view/image/2021-08-30_7_6.tiff',
                     'E:/files/view/image/2021-08-30_8_1.tiff', 'E:/files/view/image/2021-08-30_8_2.tiff',
                     'E:/files/view/image/2021-08-30_8_3.tiff', 'E:/files/view/image/2021-08-30_8_4.tiff',
                     'E:/files/view/image/2021-08-30_8_5.tiff', 'E:/files/view/image/2021-08-30_8_6.tiff',
                     'E:/files/view/image/2021-08-30_9_1.tiff', 'E:/files/view/image/2021-08-30_9_2.tiff',
                     'E:/files/view/image/2021-08-30_9_3.tiff', 'E:/files/view/image/2021-08-30_9_4.tiff',
                     'E:/files/view/image/2021-08-30_9_5.tiff', 'E:/files/view/image/2021-08-30_9_6.tiff',
                     'E:/files/view/image/2021-09-20_4_1.tiff', 'E:/files/view/image/2021-09-20_5_1.tiff',
                     'E:/files/view/image/2021-09-27_0_2.tiff', 'E:/files/view/image/2021-09-27_0_3.tiff',
                     'E:/files/view/image/2021-09-27_1_2.tiff', 'E:/files/view/image/2021-09-27_1_3.tiff',
                     'E:/files/view/image/2021-09-27_2_2.tiff', 'E:/files/view/image/2021-09-27_2_3.tiff',
                     'E:/files/view/image/2021-09-27_2_4.tiff', 'E:/files/view/image/2021-09-27_3_2.tiff',
                     'E:/files/view/image/2021-09-27_3_3.tiff', 'E:/files/view/image/2021-09-27_3_4.tiff',
                     'E:/files/view/image/2021-09-27_4_2.tiff', 'E:/files/view/image/2021-09-27_4_3.tiff',
                     'E:/files/view/image/2021-09-27_4_4.tiff', 'E:/files/view/image/2021-09-27_5_2.tiff',
                     'E:/files/view/image/2021-09-27_5_3.tiff', 'E:/files/view/image/2021-09-27_5_4.tiff',
                     'E:/files/view/image/2021-09-27_6_2.tiff', 'E:/files/view/image/2021-09-27_6_3.tiff',
                     'E:/files/view/image/2021-09-27_7_3.tiff', 'E:/files/view/image/2021-10-11_10_3.tiff',
                     'E:/files/view/image/2021-10-11_2_2.tiff', 'E:/files/view/image/2021-10-11_2_3.tiff',
                     'E:/files/view/image/2021-10-11_2_4.tiff', 'E:/files/view/image/2021-10-11_3_2.tiff',
                     'E:/files/view/image/2021-10-11_3_3.tiff', 'E:/files/view/image/2021-10-11_3_4.tiff',
                     'E:/files/view/image/2021-10-11_4_2.tiff', 'E:/files/view/image/2021-10-11_4_3.tiff',
                     'E:/files/view/image/2021-10-11_4_4.tiff', 'E:/files/view/image/2021-10-11_5_2.tiff',
                     'E:/files/view/image/2021-10-11_5_3.tiff', 'E:/files/view/image/2021-10-11_5_4.tiff',
                     'E:/files/view/image/2021-10-11_6_2.tiff', 'E:/files/view/image/2021-10-11_6_3.tiff',
                     'E:/files/view/image/2021-10-11_6_4.tiff', 'E:/files/view/image/2021-10-11_7_2.tiff',
                     'E:/files/view/image/2021-10-11_7_3.tiff', 'E:/files/view/image/2021-10-11_7_4.tiff',
                     'E:/files/view/image/2021-10-11_7_5.tiff', 'E:/files/view/image/2021-10-11_8_2.tiff',
                     'E:/files/view/image/2021-10-11_8_3.tiff', 'E:/files/view/image/2021-10-11_8_4.tiff',
                     'E:/files/view/image/2021-10-11_8_5.tiff', 'E:/files/view/image/2021-10-11_9_3.tiff',
                     'E:/files/view/image/2021-10-11_9_4.tiff', 'E:/files/view/image/2021-10-11_9_5.tiff',
                     'E:/files/view/image/2021-10-18_1_4.tiff']

translate_classes_simple = {
    '55': 1,  # ice free
    '01': 1,  # open water
    '02': 1,  # bergy water

    '10': 2,  # 1/10
    '12': 2,  # 1-2/10
    '13': 2,  # 1-3/10

    '20': 3,  # 2/10
    '23': 3,  # 2-3/10
    '24': 3,  # 2-4/10

    '30': 4,  # 3/10
    '34': 4,  # 3-4/10
    '35': 4,  # 3-5/10

    '40': 5,  # 4/10
    '45': 5,  # 4-5/10
    '46': 5,  # 4-6/10

    '50': 6,  # 5/10
    '56': 6,  # 5-6/10
    '57': 6,  # 5-7/10

    '60': 7,  # 6/10
    '67': 7,  # 6-7/10
    '68': 7,  # 6-8/10

    '70': 8,  # 7/10
    '78': 8,  # 7-8/10
    '79': 8,  # 7-9/10

    '80': 9,  # 8/10
    '89': 9,  # 8-9/10
    '81': 9,  # 8-10/10

    '90': 10,  # 9/10
    '91': 10,  # 9-10/10
    '92': 10,  # 10/10
}


def def_num(it: dict) -> int:
    if it['POLY_TYPE'] == 'L':  # land
        return 0
    if it['POLY_TYPE'] == 'W':  # water
        return 1
    if it['POLY_TYPE'] == 'N':  # no data
        return 11
    if it['POLY_TYPE'] == 'S':  # ice shelf / ice of land origin
        return 12

    return translate_classes_simple[it['CT']]  # ice – of any concentration


white_dict = dict(zip(white_list_shapes, [list() for _ in white_list_shapes]))

for f in white_list_shapes:
    shapes_list = gpd.read_file(f).to_crs('epsg:4326')
    file = fiona.open(f)
    # print(len(file))
    poly = unary_union(shapes_list['geometry'])

    dt = f.split('_')[2]
    dt = [dt[:4], dt[4:6], dt[6:8]]
    add_day = str(int(dt[2]) + 1)

    for img_file in white_list_images:
        if not '-'.join(dt) in img_file:
            continue

        sat_img = rasterio.open(img_file, 'r')
        bb, profile = sat_img.bounds, sat_img.profile
        sat_poly = box(*bb, ccw=True)

        geom_value = []
        for i in range(len(file)):
            geom_ = shapes_list.geometry[i].intersection(sat_poly)
            if geom_.area == 0:
                cnt += 1
                continue
            prop = file[i]
            geom_value.append((geom_, def_num(prop['properties'])))
        # print(geom_value)

        rasterized = features.rasterize(geom_value,
                                        out_shape=sat_img.shape,
                                        transform=sat_img.transform,
                                        all_touched=True,
                                        fill=13,  # undefined
                                        merge_alg=MergeAlg.replace,
                                        dtype=np.int16)
        # Plot raster
        if len(geom_value) > 3:
            fig, ax = plt.subplots(1, figsize=(10, 10))
            show(rasterized, ax=ax)
            plt.gca().invert_yaxis()
            plt.show()

        # print(sat_img.shape)
        # print(rasterized.shape)
        # print(img_file.split('image/')[1].split('.')[0])
        npy_name = img_file.split('image/')[1].split('.')[0]
        np.save(f'E:/files/label/{npy_name}', rasterized)

        # with rasterio.open(img_file.replace('image', 'map'), 'w', **profile) as src:
        #     src.write(rasterized)

        # if poly.intersects(sat_poly):
        #     cnt += 1
        #     print(f)
        #     print(img_file)
# print(cnt)

# for f in glob.glob(f'{reg_path}/*.shp'):  # [test_name]
#     shapes_list = gpd.read_file(f.replace('\\', '/')).to_crs('epsg:4326')
#     # poly = cascaded_union(shapes_list['geometry'])
#     poly = unary_union(shapes_list['geometry'])
#
#     dt = f.split('_')[2]
#     dt = [dt[:4], dt[4:6], dt[6:8]]
#     add_day = str(int(dt[2]) + 1)
#
#     for img_file in os.listdir(f'{view_path}/image'):
#         if not '-'.join(dt) in img_file:
#             continue
#
#         sat_img = rasterio.open(f'{view_path}/image/{img_file}', 'r')
#         bb, profile = sat_img.bounds, sat_img.profile
#         sat_poly = box(*bb, ccw=True)
#         if poly.intersects(sat_poly):
#             cnt += 1
#             white_list_shapes.append(f.replace('\\', '/'))
#             white_list_images.append(f'{view_path}/image/{img_file}')
#             # print(1)
#             # plt.plot(*sat_poly.exterior.xy)
#             # plt.plot(*poly.exterior.xy)
#             # plt.show()
# print(cnt)
# print()
# print(white_list_shapes)
# print()
# print(white_list_images)


# date1 = '-'.join(dt[:3])
# # date2 = '-'.join(dt[:2]) + '-' + (add_day if len(add_day) > 2 else '0' + add_day)
# time1 = str(int(f.split('Z')[0].split('T')[1]) - 400)
# time1 = time1[:2] + ':' + time1[2:] + ':00'
# time2 = '22:59:59'
# search_criteria = {
#     "productType": product_type,
#     "start": date1 + 'T' + time1,
#     "end": date1 + 'T' + time2,
#     "geom": poly,
#     "items_per_page": 3,
# }
#
# first_page, estimated_total_number = dag.search(**search_criteria)
# print(f"Got {estimated_total_number}.")
# nm = f.split('\\')[-1][4:].split('T')[0]
# prods_filepath = dag.serialize(first_page, filename=os.path.join(workspace, f"{nm}.geojson"))
#
# # dag.download_all(first_page)
# for j in range(len(first_page)):
#     product_down = first_page[j]
#     if 'EW' in product_down.properties["title"]:
#         continue
#     try:
#         product_path = product_down.download(extract=False)
#     except Exception:
#         pass
#
# zip_paths = [ar.replace('\\', '/') for ar in glob.glob(f'{workspace}/*.zip')]
# for zip_id in tqdm.tqdm(range(len(zip_paths))):
#     if not os.path.isfile(zip_paths[zip_id]):
#         continue
#
#     full_path = zip_paths[zip_id]
#     try:
#         product = Reader().open(full_path)
#     except:
#         continue
#
#     product_polygon = product.wgs84_extent().iloc[0].geometry
#     print(product_polygon.area)
