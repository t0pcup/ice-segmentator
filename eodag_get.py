import os
from datetime import date, timedelta
from eodag.api.core import EODataAccessGateway
from eodag import setup_logging
import sys
import glob
import matplotlib.pyplot as plt
from PIL import Image
import warnings
import numpy as np
import rasterio
import tqdm
from rasterio.features import shapes, rasterize
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import cascaded_union
from shapely.ops import unary_union
from pathlib import Path
import logging
from eoreader.reader import Reader
import eoreader.bands as bands
from eoreader.bands import VV, HH, VV_DSPK, HH_DSPK, HILLSHADE, SLOPE, to_str, VH, VH_DSPK
import geopandas as gpd
from sentinelhub.geo_utils import wgs84_to_utm, to_wgs84, get_utm_crs
from osgeo import gdal
from osgeo import osr

warnings.filterwarnings("ignore")
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__USERNAME"] = "katarina.spasenovic@omni-energy.it"
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__PASSWORD"] = "M@rkon!1997"
setup_logging(verbose=2)
# Create the workspace folder.
workspace = './dataset'
if not os.path.isdir(workspace):
    os.mkdir(workspace)

# Save the PEPS configuration file.
yaml_content = """
peps:
    download:
        outputs_prefix: "{}"
        extract: true
""".format(workspace)

with open(os.path.join(workspace, 'eodag_conf.yml'), "w") as f_yml:
    f_yml.write(yaml_content.strip())

dag = EODataAccessGateway(os.path.join(workspace, 'eodag_conf.yml'))


def RasterioGeo(bytestream, mask):
    geom_all = []
    with rasterio.open(bytestream) as dataset:
        for geom, val in shapes(mask, transform=dataset.transform):
            geom_all.append(rasterio.warp.transform_geom(dataset.crs, 'EPSG:4326', geom))  # , precision=6

        return geom_all


def eq(a, b):
    cnt = sum(int(b[i] - 5 < a[i] < b[i] + 5) for i in range(3))
    return int(cnt == 3)


def init_masks_2(colors, im_arr, th=5):
    mm = np.zeros([im_arr.shape[0], im_arr.shape[1], len(colors)])
    tmp = np.zeros([im_arr.shape[0], im_arr.shape[1], 3])

    for i, c in enumerate(colors):
        #         tmp = np.zeros([im_arr.shape[0], im_arr.shape[1], 3])
        for j in range(3):
            tmp[np.logical_and(im_arr[:, :, j] > c[j] - th, im_arr[:, :, j] < c[j] + th), j] = c[j]
        mm[np.logical_and(np.logical_and(tmp[:, :, 0] == c[0], tmp[:, :, 1] == c[1]), tmp[:, :, 2] == c[2]), i] = 1

    return mm.transpose((2, 0, 1))


def t(lst: list):
    return Polygon([(i[0], i[1]) for i in [j for j in lst[0]]])


def save_gj(p, i):
    gpd.GeoSeries(p).to_file(f"{i}.geojson", driver='GeoJSON', show_bbox=False)  # , crs="EPSG:4326"


def save_nw(poly: list, colour: str, name: list):
    dt = name.split('_')[2]
    dt = f"{dt[:4]}-{dt[4:6]}-{dt[6:]}"
    if type(poly) == Polygon:
        gdf = gpd.GeoDataFrame({'id': [0], 'geometry': poly, 'date': dt})
    else:
        gdf = gpd.GeoDataFrame({'id': list(range(len(poly))), 'geometry': poly, 'date': dt})
    gdf.to_file(f"alaskan_coast_{colour}.geojson", driver="GeoJSON")


def decorator(ar):
    return f"{ar * 10 ** 5}" + ["", "!!!"][ar * 10 ** 5 < 5]


def to_pol(i):
    return Polygon(t(i['coordinates']))


df = gpd.read_file(f'alyaska.geojson')

dataset_path = "E:\\dataset"
# zip_paths = glob.glob("dataset\*.zip")

save_path = "E:\\SAR_dataset_v2"

for image_name in tqdm.tqdm_notebook(files[88:]):  # os.listdir(path)
    dataframe = None
    conc_list = []
    min_list = []
    image = Image.open(path + image_name)
    im_arr = np.asarray(image)

    mask_date = image_name.split('_')[2]
    year = mask_date[:4]
    month = mask_date[4:6]
    day = mask_date[6:]

    date_1 = date.fromisoformat(f'{year}-{month}-{day}')
    td = timedelta(days=1)
    date_2 = date_1 + td

    date_1 = date_1.strftime('%Y-%m-%d')
    date_2 = date_2.strftime('%Y-%m-%d')

    masks = init_masks_2(list(colors.values()), im_arr, th=7)
    print(f'{date_1} : start')
    for i in range(len(colors)):  # ['orange']:
        #         print(list(colors.keys())[i])
        # mask_2d = np.array(init_mask(colors[i], im_arr))
        mask_2d = masks[i]
        mask_2d = mask_2d[0:np.asarray(image)[:, :, :2].shape[0], 0:np.asarray(image)[:, :, :2].shape[1]]

        h, w = mask_2d.shape
        mask_3d = np.zeros((h, w), dtype='uint8')
        mask_3d[mask_2d[:, :] > 0.5] = 255

        polygons = RasterioGeo(path + image_name, mask_3d)
        min_ = [[j[1] for j in p['coordinates'][0]] for p in polygons]
        min_list.append(min_)
        mp = cascaded_union(
            MultiPolygon([to_pol(p) for p in polygons if min(min(min_)) not in to_pol(p).exterior.coords.xy[1]])[:-1])

        if type(mp) == Polygon:
            conc_list.append(mp)

        else:
            conc_list.append(MultiPolygon(mp))

    if dataframe is None:
        dataframe = gpd.GeoDataFrame({'geometry': conc_list, 'date': [date_1 for i in range(7)], 'type': colors.keys()})
    else:
        dataframe = dataframe.append(
            gpd.GeoDataFrame({'geometry': conc_list, 'date': [date_1 for i in range(7)], 'type': colors.keys()}))

    print(f'{date_1} : mask done')

    board = cascaded_union(conc_list).bounds

    board = cascaded_union(conc_list).bounds

    product_type = 'S1_SAR_GRD'
    extent = {
        'lonmin': board[0],
        'lonmax': board[2],
        'latmin': board[1],
        'latmax': board[3]
    }

    products, estimated_nbr_of_results = dag.search(
        productType=product_type,
        start=date_1,
        end=date_2,
        geom=extent,
        items_per_page=500
    )

    for j in range(len(products)):
        product_down = products[j]

        if 'EW' in product_down.properties["title"]:
            continue
        try:
            product_path = product_down.download(extract=False)
        except Exception:
            pass

    zip_paths = glob.glob(f'{workspace}/*.zip')

    for zip_id in tqdm.tqdm(range(len(zip_paths))):

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

        if '1SDH' in name.split('_'):
            print(1)
            continue

        mask_date = name.split('_')[4]
        year = mask_date[:4]
        month = mask_date[4:6]
        day = mask_date[6:8]

        date_1 = date.fromisoformat(f'{year}-{month}-{day}')

        date_1 = date_1.strftime('%Y-%m-%d')

        df_date = df[df['date'] == date_1]

        mp = [df_date.iloc[i].geometry for i in range(len(df_date))]

        mp_clear = []
        for m in mp:
            if m is not None:
                mp_clear.append(m)

        mp = unary_union(mp_clear)
        pi = product_polygon.intersection(mp)
        print(f"Intersection: {pi.area}")

        if pi.area < 0.02:
            print(2)
            continue

        product_polygon = product.wgs84_extent().iloc[0].geometry

        crs = get_utm_crs(product_polygon.bounds[0], product_polygon.bounds[1])

        print(product.crs(), crs)

        min_mp_utmx, min_mp_utmy = wgs84_to_utm(mp.bounds[0], mp.bounds[1], crs)
        max_mp_utmx, max_mp_utmy = wgs84_to_utm(mp.bounds[2], mp.bounds[3], crs)

        bands = [VV, HH, VV_DSPK, HH_DSPK, VH, VH_DSPK]
        ok_bands = [band for band in bands if product.has_band(band)]

        stack = product.stack(ok_bands)

        print('Stack down')
        np_stack = stack.to_numpy()

        resolution = product.resolution
        chunk_size = int((256 / 20) * 100)

        min_mp_utmx = max(min_mp_utmx, stack.x[0])
        min_mp_utmy = max(min_mp_utmy, stack.y[-1])
        max_mp_utmx = min(max_mp_utmx, stack.x[-1])
        max_mp_utmy = min(max_mp_utmy, stack.y[0])

        if min_mp_utmx > max_mp_utmx or min_mp_utmy > max_mp_utmy:
            print(min_mp_utmx, max_mp_utmx, min_mp_utmy, max_mp_utmy)
            print(3)
            continue

        min_x = int((np.abs(stack.x - min_mp_utmx)).argmin())
        max_y = int((np.abs(stack.y - min_mp_utmy)).argmin())
        max_x = int((np.abs(stack.x - max_mp_utmx)).argmin())
        min_y = int((np.abs(stack.y - max_mp_utmy)).argmin())

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

                    xmin, ymax = to_wgs84(stack.x[x1], stack.y[y1], crs)
                    xmax, ymin = to_wgs84(stack.x[x2], stack.y[y2], crs)

                    patch_pol = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

                    inter = patch_pol.intersection(mp)

                    if inter.area < 0.0005:
                        continue

                    nx = patch.shape[1]
                    ny = patch.shape[1]

                    xres = (xmax - xmin) / float(nx)
                    yres = (ymax - ymin) / float(ny)
                    geotransform = (xmin, xres, 0, ymax, 0, -yres)

                    save_tiff = os.path.join(save_path, f'{date_1}_{sx}_{sy}')

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
                    dataframe = gpd.GeoDataFrame({'geometry': [patch_pol], 'date': date_1})
                    dataframe.to_file(f"{save_tiff}.geojson", driver='GeoJSON', show_bbox=False)

        except Exception:
            pass
    for zip_id in tqdm.tqdm(range(len(zip_paths))):
        if not os.path.isfile(zip_paths[zip_id]):
            continue

        else:
            os.remove(zip_paths[zip_id])
