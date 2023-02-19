import warnings
from shapely.geometry import Polygon
from shapely.ops import unary_union
import os
import glob
from eodag import setup_logging
import geopandas as gpd
from tqdm import tqdm, trange
from datetime import date
from eoreader.reader import Reader
from sentinelhub.geo_utils import wgs84_to_utm, to_wgs84, get_utm_crs
from osgeo import gdal
from osgeo import osr
from eoreader.bands import *
import numpy as np
import sys
import fiona

warnings.filterwarnings("ignore")
reg_path = 'C:/files/regions/2021'
# save_path = 'C:/files/data'
# workspace = 'C:/files/small_dag_img'
save_path = 'D:/data'
workspace = 'D:/dag_img'

setup_logging(0)
zip_paths = glob.glob(f'{workspace}/*1SDV*.zip')
lz = len(zip_paths)
if lz == 0:
    sys.exit('nothing to extract')

for zip_id in range(len(zip_paths)):
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

    z_dt = zip_paths[zip_id].split('2021')[1].split('_')[0]
    zip_date = '2021' + z_dt.split('T')[0]
    shapes_list = glob.glob(f'{reg_path}/*{zip_date}*.shp')
    for f in tqdm(shapes_list, ascii=True, desc=f'date {zip_date[:4]}-{zip_date[4:6]}-{zip_date[6:8]}'):
        dataset = gpd.read_file(f).to_crs('epsg:4326')
        # dt = f.split('_')[2]
        # dt = date.fromisoformat(f'{dt[:4]}-{dt[4:6]}-{dt[6:8]}')

        poly = Polygon(unary_union(dataset['geometry']))
        inter_area = product_poly.intersection(poly).area
        if inter_area == 0:
            continue

        bands = [VV, HH, VV_DSPK, HH_DSPK, VH, VH_DSPK]
        ok_bands = [band for band in bands if product.has_band(band)]
        if len(ok_bands) != 4:
            continue

        stack = product.stack(ok_bands)
        np_stack = stack.to_numpy()
        resolution = product.resolution
        chunk_size = int((256 / 20) * 100)

        min_utm = wgs84_to_utm(poly.bounds[0], poly.bounds[1], crs)
        max_utm = wgs84_to_utm(poly.bounds[2], poly.bounds[3], crs)
        min_utm = max(min_utm[0], stack.x[0]), max(min_utm[1], stack.y[-1])
        max_utm = min(max_utm[0], stack.x[-1]), min(max_utm[1], stack.y[0])

        if min_utm[0] > max_utm[0] or min_utm[1] > max_utm[1]:
            print("ERR_0")
            continue

        min_x = int((np.abs(stack.x - min_utm[0])).argmin())
        max_y = int((np.abs(stack.y - min_utm[1])).argmin())
        max_x = int((np.abs(stack.x - max_utm[0])).argmin())
        min_y = int((np.abs(stack.y - max_utm[1])).argmin())

        step_x = (max_x - min_x) // chunk_size
        step_y = (max_y - min_y) // chunk_size
        for sx in range(step_x + 1):
            for sy in range(step_y + 1):
                try:
                    reg = f.split('SGRDR')[1][:2]
                    tiff_name = f'{save_path}/{reg}_{z_dt}_{zip_id}_{sx}_{sy}'
                    if tiff_name in os.listdir(save_path):
                        continue

                    y1 = min_y + sy * chunk_size
                    y2 = min_y + (sy + 1) * chunk_size

                    x1 = min_x + sx * chunk_size
                    x2 = min_x + (sx + 1) * chunk_size
                    if sum([y1 < 0, x1 < 0, y2 >= len(stack.y), x2 >= len(stack.x)]):
                        # print("ERR_1")
                        continue

                    patch = np_stack[:, y1:y2, x1:x2]
                    # print(f"PATCH SHAPE: {patch.shape}")
                    if np.sum(np.isnan(patch)) > 20 * chunk_size * 4:
                        # print("ERR_2")
                        continue

                    x_min, y_max = to_wgs84(stack.x[x1], stack.y[y1], crs)
                    x_max, y_min = to_wgs84(stack.x[x2], stack.y[y2], crs)
                    nx, ny = patch.shape[1], patch.shape[1]

                    x_res, y_res = (x_max - x_min) / float(nx), (y_max - y_min) / float(ny)
                    geo_transform = (x_min, x_res, 0, y_max, 0, -y_res)

                    np.save(tiff_name, patch)

                    bd = 2 if len(ok_bands) == 2 else 3
                    dst_ds = gdal.GetDriverByName('GTiff').Create(f'{tiff_name}.tiff', ny, nx, bd, gdal.GDT_Byte)
                    dst_ds.SetGeoTransform(geo_transform)  # specify coords
                    srs = osr.SpatialReference()  # establish encoding
                    srs.ImportFromEPSG(4326)  # WGS84 lat/long
                    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
                    dst_ds.GetRasterBand(1).WriteArray(patch[0])  # write r-band to the raster
                    dst_ds.GetRasterBand(2).WriteArray(patch[1])  # write g-band to the raster
                    if bd == 3:
                        dst_ds.GetRasterBand(3).WriteArray(patch[2])  # write b-band to the raster
                    dst_ds.FlushCache()
                    # dst_ds = None ???
                except:
                    print(f'FAIL at {sx}-{sy}', end='')
                    pass
    os.remove(full_path)
# TODO: убрать дубликаты
