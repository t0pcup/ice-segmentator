import warnings
from shapely.ops import unary_union
from shapely.geometry import Polygon, box
import os
import glob
from eodag.api.core import EODataAccessGateway
from eodag import setup_logging
import geopandas as gpd
from tqdm import tqdm
from datetime import date, timedelta
from eoreader.reader import Reader
from sentinelhub.geo_utils import wgs84_to_utm, to_wgs84, get_utm_crs
from pathlib import Path
import logging
from eoreader.reader import Reader
import eoreader.bands as bands
from eoreader.bands import VV, HH, VV_DSPK, HH_DSPK, HILLSHADE, SLOPE, to_str, VH, VH_DSPK
from sentinelhub.geo_utils import wgs84_to_utm, to_wgs84, get_utm_crs
from osgeo import gdal
from osgeo import osr
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
save_path = 'E:/dag_img'

setup_logging(1)
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__USERNAME"] = "katarina.spasenovic@omni-energy.it"
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__PASSWORD"] = "M@rkon!1997"

os.environ["EODAG__ONDA__AUTH__CREDENTIALS__USERNAME"] = "t0pcup@yandex.ru"
os.environ["EODAG__ONDA__AUTH__CREDENTIALS__PASSWORD"] = "jL7-iq4-GBM-RPe"
dag = EODataAccessGateway()
workspace = 'E:/dataset'

if not os.path.isdir(workspace):
    os.mkdir(workspace)

yaml_content = """
peps:
    download:
        outputs_prefix: "{}"
        extract: true
""".format(workspace)

with open(f'{workspace}/eodag_conf.yml', "w") as f_yml:
    f_yml.write(yaml_content.strip())

dag = EODataAccessGateway(f'{workspace}/eodag_conf.yml')
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
    '20': 2,  # 2/10
    '23': 2,  # 2-3/10
    '24': 2,  # 2-4/10

    '30': 3,  # 3/10
    '34': 3,  # 3-4/10
    '35': 3,  # 3-5/10
    '40': 3,  # 4/10
    '45': 3,  # 4-5/10
    '46': 3,  # 4-6/10

    '50': 4,  # 5/10
    '56': 4,  # 5-6/10
    '57': 4,  # 5-7/10
    '60': 4,  # 6/10
    '67': 4,  # 6-7/10
    '68': 4,  # 6-8/10

    '70': 5,  # 7/10
    '78': 5,  # 7-8/10
    '79': 5,  # 7-9/10
    '80': 5,  # 8/10
    '89': 5,  # 8-9/10
    '81': 5,  # 8-10/10

    '90': 6,  # 9/10
    '91': 6,  # 9-10/10
    '92': 6,  # 10/10
}


def def_num(it: dict) -> int:
    trans_dict = {'L': 0, 'W': 1, 'N': 0, 'S': 7}
    try:
        return trans_dict[it['POLY_TYPE']]
    except:
        return translate_classes_simple[it['CT']]


white_dict = dict(zip(white_list_shapes, [list() for _ in white_list_shapes]))

"""NEW"""
for f in glob.glob(f'E:/files/regions/2021/*.shp'):
    dataset = gpd.read_file(f).to_crs('epsg:4326')
    dt = f.split('_')[2]
    dt = date.fromisoformat(f'{dt[:4]}-{dt[4:6]}-{dt[6:8]}')
    nm = f.split('\\')[-1][4:].split('T')[0]
    search_criteria = {
        "productType": product_type,
        "start": f'{dt}T10:00:00',
        "end": f'{dt}T23:59:59',
        "geom": None,
        "items_per_page": 500,
    }

    i = -1
    for item in [unary_union(dataset['geometry'])]:  # dataset['geometry']:
        print(0, end='')
        i += 1
        poly = Polygon(item)
        search_criteria["geom"] = poly
        # first_page, estimated_total_number = dag.search(**search_criteria)
        # if estimated_total_number == 0:
        #     continue
        #
        # # dag.download_all(first_page)
        # for elt in first_page:
        #     if '1SDH' in elt.properties["title"] or 'EW' in elt.properties["title"]:
        #         continue
        #     try:
        #         product_path = elt.download(extract=False)
        #     except:
        #         pass

        zip_paths = glob.glob(f'{workspace}/*.zip')
        if len(zip_paths) == 0:
            continue

        for zip_id in tqdm(range(len(zip_paths))):
            if not os.path.isfile(zip_paths[zip_id]):
                continue

            full_path = os.path.join(dataset_path, zip_paths[zip_id])
            reader = Reader()
            try:
                product = reader.open(full_path)
            except:
                continue

            product_polygon = product.wgs84_extent().iloc[0].geometry
            name = os.path.basename(full_path)
            print(f'{name}')

            # mask_date = name.split('_')[4]
            # year = mask_date[:4]
            # month = mask_date[4:6]
            # day = mask_date[6:8]
            #
            # date_1 = date.fromisoformat(f'{year}-{month}-{day}')
            # date_1 = date_1.strftime('%Y-%m-%d')
            # df_date = df[df['date'] == date_1]
            # mp = [df_date.iloc[i].geometry for i in range(len(df_date))]

            pi = product_polygon.intersection(poly)
            print(f"Intersection: {pi.area}")
            if pi.area < 0.02:
                print(2)
                continue

            product_polygon = product.wgs84_extent().iloc[0].geometry
            crs = get_utm_crs(product_polygon.bounds[0], product_polygon.bounds[1])
            print(product.crs(), crs)

            min_utm_x, min_utm_y = wgs84_to_utm(poly.bounds[0], poly.bounds[1], crs)
            max_utm_x, max_utm_y = wgs84_to_utm(poly.bounds[2], poly.bounds[3], crs)

            bands = [VV, HH, VV_DSPK, HH_DSPK, VH, VH_DSPK]
            ok_bands = [band for band in bands if product.has_band(band)]

            stack = product.stack(ok_bands)

            print('Stack down')
            np_stack = stack.to_numpy()

            resolution = product.resolution
            chunk_size = int((256 / 20) * 100)

            min_utm_x = max(min_utm_x, stack.x[0])
            min_utm_y = max(min_utm_y, stack.y[-1])
            max_utm_x = min(max_utm_x, stack.x[-1])
            max_utm_y = min(max_utm_y, stack.y[0])

            if min_utm_x > max_utm_x or min_utm_y > max_utm_y:
                print(min_utm_x, max_utm_x, min_utm_y, max_utm_y)
                print(3)
                continue

            min_x = int((np.abs(stack.x - min_utm_x)).argmin())
            max_y = int((np.abs(stack.y - min_utm_y)).argmin())
            max_x = int((np.abs(stack.x - max_utm_x)).argmin())
            min_y = int((np.abs(stack.y - max_utm_y)).argmin())

            step_x = (max_x - min_x) // chunk_size
            step_y = (max_y - min_y) // chunk_size

            print(f'Step x: {step_x}; Step y: {step_y}')

            try:

                for sx in range(step_x + 1):
                    for sy in range(step_y + 1):
                        y1 = min_y + sy * chunk_size
                        y2 = min_y + (sy + 1) * chunk_size

                        x1 = min_x + sx * chunk_size
                        x2 = min_x + (sx + 1) * chunk_size
                        if y1 < 0 or x1 < 0:
                            continue
                        if y2 >= len(stack.y) or x2 >= len(stack.x):
                            continue
                        patch = np_stack[:, y1:y2, x1:x2]

                        if np.sum(np.isnan(patch)) > 20 * chunk_size * 4:
                            continue

                        x_min, y_max = to_wgs84(stack.x[x1], stack.y[y1], crs)
                        x_max, y_min = to_wgs84(stack.x[x2], stack.y[y2], crs)

                        patch_pol = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])
                        inter = patch_pol.intersection(poly)

                        if inter.area < 0.0005:
                            continue

                        nx = patch.shape[1]
                        ny = patch.shape[1]

                        x_res = (x_max - x_min) / float(nx)
                        y_res = (y_max - y_min) / float(ny)
                        geotransform = (x_min, x_res, 0, y_max, 0, -y_res)
                        save_tiff = f'{save_path}/{dt}_{i}_{sx}_{sy}'

                        dst_ds = gdal.GetDriverByName('GTiff').Create(f'{save_tiff}.tiff', ny, nx, 3, gdal.GDT_Byte)
                        dst_ds.SetGeoTransform(geotransform)  # specify coords
                        srs = osr.SpatialReference()  # establish encoding
                        srs.ImportFromEPSG(4326)  # WGS84 lat/long
                        dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
                        dst_ds.GetRasterBand(1).WriteArray(patch[0])  # write r-band to the raster
                        dst_ds.GetRasterBand(2).WriteArray(patch[1])  # write g-band to the raster
                        dst_ds.GetRasterBand(3).WriteArray(patch[2])  # write b-band to the raster
                        dst_ds.FlushCache()
                        np.save(save_tiff, patch)
                        dataframe = gpd.GeoDataFrame({'geometry': [patch_pol], 'date': dt})
                        dataframe.to_file(f"{save_tiff}.geojson", driver='GeoJSON', show_bbox=False)
            except:
                pass
        for zip_id in tqdm(range(len(zip_paths))):
            if not os.path.isfile(zip_paths[zip_id]):
                continue
            else:
                os.remove(zip_paths[zip_id])
"""NEW"""

"""BEGIN"""
# for f in white_list_shapes:
#     shapes_list = gpd.read_file(f).to_crs('epsg:4326')
#     file = fiona.open(f)
#     # print(len(file))
#     poly = unary_union(shapes_list['geometry'])
#
#     dt = f.split('_')[2]
#     dt = [dt[:4], dt[4:6], dt[6:8]]
#     add_day = str(int(dt[2]) + 1)
#
#     for img_file in white_list_images:
#         if not '-'.join(dt) in img_file:
#             continue
#
#         sat_img = rasterio.open(img_file, 'r')
#         bb, profile = sat_img.bounds, sat_img.profile
#         sat_poly = box(*bb, ccw=True)
#
#         geom_value = []
#         for i in range(len(file)):
#             geom_ = shapes_list.geometry[i].intersection(sat_poly)
#             if geom_.area == 0:
#                 cnt += 1
#                 continue
#             prop = file[i]
#             geom_value.append((geom_, def_num(prop['properties'])))
#         # print(geom_value)
#
#         rasterized = features.rasterize(geom_value,
#                                         out_shape=sat_img.shape,
#                                         transform=sat_img.transform,
#                                         all_touched=True,
#                                         fill=0,  # undefined
#                                         merge_alg=MergeAlg.replace,
#                                         dtype=np.int16)
#
#         if rasterized.all() == 0:
#             continue
#         if len(geom_value) > 5:
#             fig, ax = plt.subplots(1, figsize=(10, 10))
#             show(rasterized, ax=ax)
#             plt.gca().invert_yaxis()
#             plt.show()
#
#         # print(sat_img.shape)
#         # print(rasterized.shape)
#         # print(img_file.split('image/')[1].split('.')[0])
#         npy_name = img_file.split('image/')[1].split('.')[0]
#         np.save(f'E:/files/label/{npy_name}', rasterized)
"""END"""

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
