import shapely
import PIL
from PIL import Image
import numpy as np
from varname import nameof
import warnings
import os
from os import listdir
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
from shapely.ops import cascaded_union, unary_union
import rasterio
from rasterio.features import shapes, rasterize
# from geojson import MultiPolygon, Polygon todo
from tqdm import tqdm

warnings.filterwarnings("ignore")
colors = {
    'blue': [0, 0, 255],
    'green': [0, 255, 0],
    'yellow': [255, 255, 0],
    'red': [255, 0, 0],
}
conc_list = []


def RasterioGeo(bytestream, mask):
    geom_all = []
    with rasterio.open(bytestream) as dataset:
        for geom, val in shapes(mask, transform=dataset.transform):
            if val != 0:
                geom_all.append(rasterio.warp.transform_geom(dataset.crs, 'EPSG:4326', geom))  # , precision=6
        return geom_all


def init_mask(color, im_):
    a = np.array([np.zeros(len(im_[0])) for _ in im_])
    flag = False

    for k in range(len(im_)):
        for j in range(len(im_[k])):
            if not flag and list(im_[k, j]) == color:
                flag = True
            a[k, j] = int(list(im_[k, j]) == color)

    return a


def eq(a, b):
    cnt = sum(int(b[i_] - 5 < a[i_] < b[i_] + 5) for i_ in range(3))
    return int(cnt == 3)


def init_masks(colors_, im_):
    masks_ = [np.array([np.zeros(len(im_[0])) for _ in im_]) for c in colors_]

    for q in range(len(im_)):
        for j in range(len(im_[q])):
            for k in range(len(colors_)):
                (masks_[k])[q, j] = eq(list(im_[q, j]), colors_[k])

    return masks_


def t(lst: list):
    # print([j for j in lst[0]])
    # return Polygon([j for j in lst[0]])
    # print(lst)
    # print([j for j in lst[0]])
    # print([(k[0], k[1]) for k in [j for j in lst[0]]])
    return shapely.Polygon([(k[0], k[1]) for k in [j for j in lst[0]]])


def save_gj(p, c):
    # p = shapely.MultiPolygon(p)
    # print(p)
    # print(type(p))
    sp = path.replace('p/', '')
    gpd.GeoSeries(p).to_file(f"{sp}{c}.geojson", driver='GeoJSON', show_bbox=False)  # , crs="EPSG:4326"


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


def to_pol(j):
    return t(j['coordinates'])
    # return shapely.Polygon(t(j['coordinates']))


path = 'D:/M/10-2+/p/'
for image_name in tqdm(os.listdir(path)[:1]):
    image = rasterio.open(path + image_name, 'r')  # Image.open(path + image_name)
    im_arr = np.asarray(image.read()).transpose((1, 2, 0))

    masks = [np.array(m) for m in init_masks(list(colors.values()), im_arr)]  # each shape is 1280,1280
    print('masks initiated')
    for i in range(len(colors)):  # ['orange']:
        # print(list(colors.keys())[i])

        # mask_2d = np.array(init_mask(colors[i], im_arr))
        mask_2d = masks[i]
        # mask_2d = mask_2d[0:np.asarray(image)[:, :, :2].shape[0], 0:np.asarray(image)[:, :, :2].shape[1]] todo

        h, w = mask_2d.shape
        mask_3d = np.zeros((h, w), dtype='uint8')
        mask_3d[mask_2d[:, :] > 0.5] = 255
        if np.sum(mask_3d) == 0:
            print('no such colour: ' + list(colors.keys())[i])
            continue

        Image.fromarray(np.uint8(mask_3d)).show()

        polygons = RasterioGeo(path + image_name, mask_3d)
        pol_list = [to_pol(p) for p in polygons]
        # # print([p.envelope.area for p in pol_list])
        # areas = [p.area for p in pol_list]
        # b_areas = [p.envelope.area for p in pol_list]
        # mxb = max(b_areas)
        # mxb_index = b_areas.index(mxb)
        # mxa = max(areas)
        # mxa_index = areas.index(mxa)
        # # print(mxa_index == mxb_index)
        # # print(mxb, mxa)
        # # print(mxb_index, mxa_index)
        #
        # if mxa_index == mxb_index:
        #     pol_list = [pol_list[p] for p in range(len(pol_list)) if p != mxb_index]
        # else:
        #     pol_list = [pol_list[p] for p in range(len(pol_list)) if p != mxa_index]

        if len(pol_list) == 0:
            print('no suitable polygons left')
            continue
        # print(pol_list)
        # print(len(pol_list))
        # print('pol_list', pol_list)
        multi_from_list = shapely.MultiPolygon(pol_list)
        # mp = unary_union(multi_from_list)
        # print('mp', mp)
        mp = multi_from_list
        save_gj(mp, image_name.split('.')[0] + '_' + list(colors.keys())[i])
        # save_gj(cascaded_union(mp), image_name.split('.')[0] + '_' + list(colors.keys())[i])

        # # min_ = [[j[0] for j in p['coordinates'][0]] for p in polygons]
        # # mp = cascaded_union(
        # #     MultiPolygon([to_pol(p) for p in polygons if min(min(min_)) not in to_pol(p).exterior.coords.xy[0]])[:-1])
        # if type(mp) == Polygon:
        #     conc_list.append((mp, list(colors.keys())[i], image_name.split('.')[0]))
        # else:
        #     conc_list.append(([MultiPolygon(mp)], list(colors.keys())[i], image_name.split('.')[0]))
        #
        # save_gj(MultiPolygon(cascaded_union(mp[:-1])), i)
