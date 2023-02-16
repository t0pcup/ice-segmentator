import sys
import warnings
from shapely.ops import unary_union
from shapely.geometry import Polygon, box
import os
import glob
import snap
import sertit
import geopandas
from eodag.api.core import EODataAccessGateway
from eodag import setup_logging
import geopandas as gpd
from tqdm import tqdm, trange
from datetime import date, timedelta
from pathlib import Path
import logging
from eoreader.reader import Reader
from eoreader.bands import *
import eoreader.bands as bands
from eoreader.bands import VV, HH, VV_DSPK, HH_DSPK, HILLSHADE, SLOPE, to_str, VH, VH_DSPK
from sentinelhub.geo_utils import wgs84_to_utm, to_wgs84, get_utm_crs
from osgeo import gdal
from osgeo import osr
from eoreader.bands import *
import matplotlib.pyplot as plt
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.plot import show
import numpy as np
import fiona

warnings.filterwarnings("ignore")
reg_path = 'C:/files/regions/2021'
save_path = 'C:/files/data'
workspace = 'D:/dag_img'

setup_logging(verbose=1, no_progress_bar=True)
# os.environ["EODAG__PEPS__AUTH__CREDENTIALS__USERNAME"] = "katarina.spasenovic@omni-energy.it"
# os.environ["EODAG__PEPS__AUTH__CREDENTIALS__PASSWORD"] = "M@rkon!1997"
# dag = EODataAccessGateway()

# if not os.path.isdir(workspace):
#     os.mkdir(workspace)

# yaml_content = """
# peps:
#     download:
#         outputs_prefix: "{}"
#         extract: true
# """.format(workspace)
#
# with open(f'{workspace}/eodag_conf.yml', "w") as f_yml:
#     f_yml.write(yaml_content.strip())

# dag = EODataAccessGateway(f'{workspace}/eodag_conf.yml')
# product_type = 'S1_SAR_GRD'
# dag.set_preferred_provider("peps")
# cnt = 0

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
    # undefined / land / no data => zero
    trans_dict = {'L': 0, 'W': 1, 'N': 0, 'S': 7}
    try:
        return trans_dict[it['POLY_TYPE']]
    except:
        return translate_classes_simple[it['CT']]


def verify(file: str):
    schema_dataset = fiona.open(file)
    # get water, ice, shelf; ignore land and no data: ['L', 'N']
    lst, idx = ['W', 'I', 'S'], []
    my_ind = -1
    for obj in schema_dataset:
        my_ind += 1
        if obj['properties']['POLY_TYPE'] in lst:
            idx.append(my_ind)
    return idx


zip_paths = glob.glob(f'{workspace}/*.zip')
if len(zip_paths) == 0:
    sys.exit('nothing to extract')

for f in glob.glob('C:/files/regions/2021/*.shp'):
    indexes = verify(f)
    dataset = gpd.read_file(f).to_crs('epsg:4326')
    nm = f.split("\\")[-1][4:].split("T")[0][5:7]
    dt = f.split('_')[2]
    dt = date.fromisoformat(f'{dt[:4]}-{dt[4:6]}-{dt[6:8]}')
    desc = f'[{len(dataset.iloc[indexes])}/{len(dataset)}] {nm} {dt}'

    for i, item in enumerate(dataset['geometry'].iloc[indexes]):
        poly = Polygon(item)

        for zip_id in trange(len(zip_paths), desc=desc, ascii=True):
            if not os.path.isfile(zip_paths[zip_id]):
                continue
            full_path = os.path.join(workspace, zip_paths[zip_id])
            reader = Reader()
            try:
                product = reader.open(full_path)
            except:
                continue

            product_poly = product.wgs84_extent().iloc[0].geometry
            name = os.path.basename(full_path)

            crs = get_utm_crs(product_poly.bounds[0], product_poly.bounds[1])
            inter_area = product_poly.intersection(poly).area
            if inter_area == 0:
                continue

            print(f"Intersection: {inter_area:.6f}", product.crs(), crs)

            bands = [VV, HH, VV_DSPK, HH_DSPK, VH, VH_DSPK]
            ok_bands = [band for band in bands if product.has_band(band)]
            stack = product.stack(ok_bands)

            np_stack = stack.to_numpy()
            resolution = product.resolution
            chunk_size = int((256 / 20) * 100)

            min_utm = wgs84_to_utm(poly.bounds[0], poly.bounds[1], crs)
            max_utm = wgs84_to_utm(poly.bounds[2], poly.bounds[3], crs)
            min_utm = max(min_utm[0], stack.x[0]), max(min_utm[1], stack.y[-1])
            max_utm = min(max_utm[0], stack.x[-1]), min(max_utm[1], stack.y[0])

            if min_utm[0] > max_utm[0] or min_utm[1] > max_utm[1]:
                continue

            min_x = int((np.abs(stack.x - min_utm[0])).argmin())
            max_y = int((np.abs(stack.y - min_utm[1])).argmin())
            max_x = int((np.abs(stack.x - max_utm[0])).argmin())
            min_y = int((np.abs(stack.y - max_utm[1])).argmin())

            step_x = (max_x - min_x) // chunk_size
            step_y = (max_y - min_y) // chunk_size

            try:
                for sx in range(step_x + 1):
                    for sy in range(step_y + 1):
                        y1 = min_y + sy * chunk_size
                        y2 = min_y + (sy + 1) * chunk_size

                        x1 = min_x + sx * chunk_size
                        x2 = min_x + (sx + 1) * chunk_size
                        if sum([y1 < 0, x1 < 0, y2 >= len(stack.y), x2 >= len(stack.x)]):
                            continue

                        patch = np_stack[:, y1:y2, x1:x2]

                        if np.sum(np.isnan(patch)) > 20 * chunk_size * 4:
                            continue

                        x_min, y_max = to_wgs84(stack.x[x1], stack.y[y1], crs)
                        x_max, y_min = to_wgs84(stack.x[x2], stack.y[y2], crs)
                        nx, ny = patch.shape[1], patch.shape[1]

                        x_res, y_res = (x_max - x_min) / float(nx), (y_max - y_min) / float(ny)
                        geo_transform = (x_min, x_res, 0, y_max, 0, -y_res)
                        tiff_name = f'{save_path}/{dt}_{i}_{sx}_{sy}'

                        dst_ds = gdal.GetDriverByName('GTiff').Create(f'{tiff_name}.tiff', ny, nx, 3, gdal.GDT_Byte)
                        dst_ds.SetGeoTransform(geo_transform)  # specify coords
                        srs = osr.SpatialReference()  # establish encoding
                        srs.ImportFromEPSG(4326)  # WGS84 lat/long
                        dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
                        dst_ds.GetRasterBand(1).WriteArray(patch[0])  # write r-band to the raster
                        dst_ds.GetRasterBand(2).WriteArray(patch[1])  # write g-band to the raster
                        dst_ds.GetRasterBand(3).WriteArray(patch[2])  # write b-band to the raster
                        dst_ds.FlushCache()
                        np.save(tiff_name, patch)
                        # dataframe = gpd.GeoDataFrame({'geometry': [patch_pol], 'date': str(dt)})
                        # dataframe.to_file(f"{tiff_name}.geojson", driver='GeoJSON', show_bbox=False)
            except:
                pass
