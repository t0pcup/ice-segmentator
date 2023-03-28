import shapely
import rasterio
import warnings
import numpy as np
import geopandas as gpd

from rasterio.features import shapes
from shapely import wkt


def RasterioGeo(bytestream, mask):
    geom_all = []
    with rasterio.open(bytestream) as dataset:
        for geom, val in shapes(mask, transform=dataset.transform):
            if val != 0.0:
                geom_all.append(rasterio.warp.transform_geom(dataset.crs, 'EPSG:3857', geom))  # , precision=6
        return geom_all


def t(lst: list):
    return shapely.Polygon([(k[0], k[1]) for k in [j for j in lst[0]]])


def save_gj(url, path, p, c):
    sp = path.replace('p/', '')
    a = gpd.GeoSeries(p).to_json()

    p2 = wkt.loads(url.split('GEOMETRY=')[1])
    b = gpd.GeoSeries(p2).to_json()
    result = b.split("bbox\": [")[1].split("]")[0]

    gpd.GeoSeries(p).to_file(f"{sp}{c}.json", driver='GeoJSON', show_bbox=False)  # , crs="EPSG:4326"
    return a, result


def to_pol(j):
    return t(j['coordinates'])


def vectorize(url):
    path = ''
    warnings.filterwarnings("ignore")
    colors = {
        'white': 1.0,
    }
    for image_name in ['src/main/python/images/outputs/output_water.tiff']:
        image = rasterio.open(path + image_name, 'r')
        im_arr = np.asarray(image.read()).transpose((1, 2, 0))

        masks = [im_arr.reshape(im_arr.shape[:2])]
        for i in range(len(colors)):
            mask_2d = masks[i]

            h, w = mask_2d.shape
            mask_3d = np.zeros((h, w), dtype='uint8')
            mask_3d[mask_2d[:, :] > 0.5] = 255
            if np.sum(mask_3d) == 0:
                print('no such colour: ' + list(colors.keys())[i])
                continue

            # Image.fromarray(np.uint8(mask_3d)).show()

            polygons = [to_pol(p) for p in RasterioGeo(path + image_name, mask_3d)]
            if len(polygons) == 0:
                print('no suitable polygons left')
                continue

            mp = shapely.MultiPolygon(polygons)
            return save_gj(url, path, mp, image_name.split('.')[0] + '_' + list(colors.keys())[i])
