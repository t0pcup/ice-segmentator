# from matplotlib.colors import ListedColormap
# from PIL import Image
# import pandas as pd
# import sys
# import matplotlib.pyplot as plt
# from PIL import Image
# import warnings
# import numpy as np
# import fiona
# import tarfile
# import rasterio
# import tqdm
# from rasterio.features import shapes, rasterize
# from shapely.geometry import Polygon, MultiPolygon
# from shapely.ops import cascaded_union, unary_union
# from pathlib import Path
# import logging
# from eoreader.reader import Reader
# import eoreader.bands as bands
# from eoreader.bands import VV, HH, VV_DSPK, HH_DSPK, HILLSHADE, SLOPE, to_str, VH, VH_DSPK
# import geopandas as gpd
# from sentinelhub.geo_utils import wgs84_to_utm, to_wgs84, get_utm_crs
# from osgeo import gdal
# from osgeo import osr
# import os
# import matplotlib.image as mpimg
# import glob
# from datetime import date, timedelta
# from eodag.api.core import EODataAccessGateway
# from eodag import setup_logging
# import rasterio
# from rasterio.plot import show
# from rasterio.enums import ColorInterp
# from rasterio.plot import show
# from rasterio.windows import Window
# from pyproj import Transformer
#
# # import tensorflow as tf
# # from tensorflow.keras.preprocessing import image_dataset_from_directory
# # from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
# warnings.filterwarnings("ignore")
# reg_path = 'E:/dafuck/regions'
# view_path = 'E:/dafuck/view'
#
#
# def run_through_dataset(dataset):
#     print(dataset.schema)
#     ps = []
#     transformer_3857_4326 = Transformer.from_crs("epsg:3857", "epsg:4326")
#     while True:
#         try:
#             nxt = dataset.next()
#             # print(nxt)
#             geom = nxt['geometry']
#             # # print(nxt)
#             # # print(geom['coordinates'])
#             p_3857 = get_poly_by_type(geom)  # TODO: это не 3857, нужно переводить из кастомных
#             # x, y = p_3857.exterior.xy
#             # print(f'min/max\tx={min(x)}/{max(x)}\ty={min(y)}/{max(y)}')
#
#             proj_points = []
#             for point in list(p_3857.exterior.coords):
#                 x = point[0]
#                 y = point[1]
#                 x, y = transformer_3857_4326.transform(y, x)
#                 proj_points.append((y, x))
#
#             p = p_4326 = Polygon(proj_points)
#             # p = p_3857
#
#             # print(int(poly.area // 1000), end='\t')
#             # shapes[shp_file].append(poly)
#             if len(ps) < 1:
#                 x, y = p.exterior.xy
#                 print(p.area)
#                 print(f'min/max\tx={min(x)}/{max(x)}\ty={min(y)}/{max(y)}')
#                 # plt.plot(x, y)
#                 # plt.title(f'min/max. x={min(x)}/{max(x)}. y={min(y)}/{max(y)}')
#                 # plt.show()
#             ps.append(p)
#
#         except Exception:
#             # print(len(ps))
#             # return MultiPolygon(ps)
#             return cascaded_union(ps)
#
#             # print('end')
#             # # print(shapes[shp_file])
#             # continue
#             # # pass
#
#
# def run_through_shapes(suitable, bbox):
#     for shp_file in suitable:
#         # print(shp_file)
#         shape = fiona.open(f'{reg_path}/{shp_file}')
#         # shapes[shp_file] = []
#         # print(shape.schema)
#
#         mp = run_through_dataset(shape)
#         if mp.intersects(bbox):
#             print(mp.intersects(bbox))
#             print(mp.contains(bbox))
#             return mp
#     return -1
#
#
# def get_poly_by_type(g):
#     if g['type'] == 'Polygon':
#         return Polygon(g['coordinates'][0])
#     return MultiPolygon(g['coordinates'])
#
#
# """raster masks making"""
# # for shp_file in glob.glob(f'{reg_path}/*.shp')[:1]:
# # for file in glob.glob(f'{path}/data/*.tiff'):
# #     name = file.split('\\')[1].split('.')[0]
# #     date = name.split('_')[0]
# #     sat_img = rasterio.open(f'{path}/data/{name}.tiff', 'r')
# #     bb = sat_img.bounds
# #     # print(bb[0])  # left
# #     # print(bb[1])  # bottom
# #     # print(bb[2])  # right
# #     # print(bb[3])  # top
# #     coords = [(bb[0], bb[3]), (bb[2], bb[3]), (bb[2], bb[1]), (bb[0], bb[1])]
# #     img_bbox, profile = Polygon(coords), sat_img.profile
# #
# #     suitable_shapes = [i.split('\\')[1] for i in glob.glob(f"{reg_path}/*{date.replace('-', '')}*.shp")]
# #     if len(suitable_shapes) == 0:
# #         continue
# #     print(f'name: {name}\tdate: {date}')
# #     # print(suitable_shapes)
# #
# #     poly = run_through_shapes(suitable_shapes, img_bbox)
# #     if poly == -1:
# #         continue
# #     print(poly)
# #     raise Exception
#
# setup_logging(1)
#
# workspace = os.path.join(view_path, "/quicklooks")
# if not os.path.isdir(workspace):
#     os.mkdir(workspace)
#
# os.environ["EODAG__PEPS__AUTH__CREDENTIALS__USERNAME"] = "t0pcup"
# os.environ["EODAG__PEPS__AUTH__CREDENTIALS__PASSWORD"] = "20010608Kd!"
# # dag = EODataAccessGateway()
# # Save the PEPS configuration file.
# yaml_content = """
# peps:
#     download:
#         outputs_prefix: "{}"
#         extract: true
# """.format(workspace)
#
# with open(os.path.join(workspace, 'eodag_conf.yml'), "w") as f_yml:
#     f_yml.write(yaml_content.strip())
#
# dag = EODataAccessGateway(os.path.join(workspace, 'eodag_conf.yml'))
# product_type = 'S1_SAR_GRD'
# dag.set_preferred_provider("peps")
#
# for f in glob.glob(f'{reg_path}/*WA*.shp')[:1]:
#     # file_name = glob.glob(f'{reg_path}/*.shp')[2].replace('\\', '/')
#     shapes_list = fiona.open(f.replace('\\', '/'))
#     poly = run_through_dataset(shapes_list)
#     # print(poly.exterior.coords.xy)
#     # print(poly.bounds)  # lon, lat, LON, LAT
#
#     dt = f.split('_')[2]
#     dt = list(map(int, [dt[:4], dt[4:6], dt[6:8], dt[9:11]]))
#     new_dt = list(map(str, [dt[0], dt[1], dt[2]+1, dt[3] + 3]))
#     dt[3] -= 5
#     dt = list(map(str, dt))
#
#     # plt.plot(*poly.exterior.xy)
#     # plt.title(f.split('cis_')[1].split('Z')[0])
#     # plt.show()
#
#     print('-'.join(dt[:3]), '\t', '-'.join(new_dt[:3]))
#     # crs = "epsg:4326"
#     crs = "epsg:3857"
#     search_criteria = {
#         "productType": product_type,
#         # "start": '-'.join(dt[:3]),
#         # "end": '-'.join(new_dt[:3]),
#         # "start": '-'.join(dt[:3]) + "T" + dt[3] + ":0:0",
#         # "end": '-'.join(new_dt[:3]) + "T" + new_dt[3] + ":59:59",
#         "geom": poly,
#         "items_per_page": 5,
#         # 'crs': crs
#     }
#
#     first_page, estimated_total_number = dag.search(**search_criteria)
#     print(f"Got {len(first_page)} and an estimated total number of {estimated_total_number}.")
#     nm = f.split('\\')[-1][3:].split('_p')[0]
#     filtered_prods_filepath = dag.serialize(first_page, filename=os.path.join(workspace, f"{nm}.geojson"))
#
#     fig = plt.figure(figsize=(10, 8))
#     for i, product in enumerate(first_page, start=1):
#         # This line takes care of downloading the quicklook
#         quicklook_path = product.get_quicklook(base_dir=workspace)
#
#         img = mpimg.imread(quicklook_path)
#         # ax = fig.add_subplot(3, 4, i)
#         # ax.set_title(i)
#         plt.imshow(img)
#         plt.tight_layout()
#         plt.show()
#
# """ ---------------------------------------------------------------------------------- """
# # def map_mask(mask, lib):
# #     new_mask = mask.copy()
# #     for key, val in lib.items():  # map the elements of the array to their new values according to the library
# #         new_mask[mask == key] = val
# #     return new_mask
# #
# #
# # ice_colors = 7
# # jet = plt.get_cmap('jet', ice_colors)
# # newcolors = jet(np.linspace(0, 1, ice_colors))
# # black, white = np.array([[0, 0, 0, 1]]), np.array([[1, 1, 1, 1]])
# # newcolors = np.concatenate((newcolors, black), axis=0)  # land will be black
# # cmap = ListedColormap(newcolors)
# # mask_lib = {55: 0, 1: 0, 2: 0, 10: 1, 12: 1, 13: 1, 20: 1, 23: 1, 24: 2, 30: 2, 34: 2, 35: 2, 40: 2, 45: 2, 46: 3,
# #             50: 3, 56: 3, 57: 3, 60: 3, 67: 3, 68: 4, 70: 4, 78: 4, 79: 4, 80: 4, 89: 4, 81: 5, 90: 5, 91: 5, 92: 6,
# #             100: 7, 99: 7}
# # # rewrite png masks to npy
# # path = 'C:/Users/mini-/Downloads/archive/Masks/'
# # label_path = 'E:/dafuck/label/'
# # uq = []
# # for file in os.listdir("C:/Users/mini-/Downloads/archive/Masks"):  # ['P1-2018061817-mask.png']
# #     img = Image.open(path + file)
# #     arr = map_mask(np.asarray(img), mask_lib)
# #     np.save(label_path + file.split('-mask')[0] + '.npy', arr)
#
# # os.environ["EODAG__PEPS__AUTH__CREDENTIALS__USERNAME"] = "katarina.spasenovic@omni-energy.it"
# # os.environ["EODAG__PEPS__AUTH__CREDENTIALS__PASSWORD"] = "M@rkon!1997"
# # setup_logging(verbose=2)
# #
# # # Create the workspace folder.
# # workspace = './dataset'
# # if not os.path.isdir(workspace):
# #     os.mkdir(workspace)
# #
# # # Save the PEPS configuration file.
# # yaml_content = """
# # peps:
# #     download:
# #         outputs_prefix: "{}"
# #         extract: true
# # """.format(workspace)
# #
# # with open(os.path.join(workspace, 'eodag_conf.yml'), "w") as f_yml:
# #     f_yml.write(yaml_content.strip())
# #
# # dag = EODataAccessGateway(os.path.join(workspace, 'eodag_conf.yml'))
# #
# #
# # def RasterioGeo(bytestream, mask):
# #     geom_all = []
# #     with rasterio.open(bytestream) as dataset:
# #         for geom, val in shapes(mask, transform=dataset.transform):
# #             geom_all.append(rasterio.warp.transform_geom(dataset.crs, 'EPSG:4326', geom))  # , precision=6
# #
# #         return geom_all
# #
# #
# # def eq(a, b):
# #     cnt = sum(int(b[i] - 5 < a[i] < b[i] + 5) for i in range(3))
# #     return int(cnt == 3)
# #
# #
# # def init_masks_2(colors_, im_arr, th=5):
# #     mm = np.zeros([im_arr.shape[0], im_arr.shape[1], len(colors_)])
# #     tmp = np.zeros([im_arr.shape[0], im_arr.shape[1], 3])
# #
# #     for i, c in enumerate(colors_):
# #         # tmp = np.zeros([im_arr.shape[0], im_arr.shape[1], 3])
# #         for j in range(3):
# #             tmp[np.logical_and(im_arr[:, :, j] > c[j] - th, im_arr[:, :, j] < c[j] + th), j] = c[j]
# #         mm[np.logical_and(np.logical_and(tmp[:, :, 0] == c[0], tmp[:, :, 1] == c[1]), tmp[:, :, 2] == c[2]), i] = 1
# #
# #     mm = mm.transpose((2, 0, 1))
# #     return mm
# #
# #
# # def t(lst: list):
# #     return Polygon([(i[0], i[1]) for i in [j for j in lst[0]]])
# #
# #
# # def save_gj(p, i):
# #     gpd.GeoSeries(p).to_file(f"{i}.geojson", driver='GeoJSON', show_bbox=False)  # , crs="EPSG:4326"
# #
# #
# # def save_nw(poly: list, colour: str, name: list):
# #     dt = name.split('_')[2]
# #     dt = f"{dt[:4]}-{dt[4:6]}-{dt[6:]}"
# #     if type(poly) == Polygon:
# #         gdf = gpd.GeoDataFrame({'id': [0], 'geometry': poly, 'date': dt})
# #     else:
# #         gdf = gpd.GeoDataFrame({'id': list(range(len(poly))), 'geometry': poly, 'date': dt})
# #     gdf.to_file(f"alaskan_coast_{colour}.geojson", driver="GeoJSON")
# #
# #
# # def decorator(ar):
# #     return f"{ar * 10 ** 5}" + ["", "!!!"][ar * 10 ** 5 < 5]
# #
# #
# # def to_pol(q):
# #     return Polygon(t(q['coordinates']))
# #
# #
# # colors = {
# #     'blue': [148, 205, 255],
# #     'green': [140, 255, 162],
# #     'yellow': [255, 255, 0],
# #     'orange': [255, 123, 0],
# #     'red': [255, 0, 0],
# #     # 'grey': [148, 147, 148],
# #     'pink': [247, 214, 255],
# #     'purple': [255, 97, 255]
# # }
# #
# # grid = gpd.read_file('E:/dafuck/view/grid_sea_ice.gpkg')
# # path = 'E:/dafuck/'
# # png_names = os.listdir('E:/dafuck/label/')
# # # print(grid.shape)  # .iloc[0]
# #
# # for line in range(grid.shape[0]):  # os.listdir(path)
# #     print(grid.iloc[line])
# #     dataframe = None
# #     conc_list = []
# #     min_list = []
# #
# #     mask_date = png_names[line][0:-4]  # .split('.')[0].split('-')
# #     print(mask_date)
# #     print(png_names[line])
# #     year = mask_date[:4]
# #     month = mask_date[4:6]
# #     day = mask_date[6:]
# #
# #     date_1 = date.fromisoformat(f'{year}-{month}-{day}')
# #     td = timedelta(days=1)
# #     date_2 = date_1 + td
# #
# #     date_1 = date_1.strftime('%Y-%m-%d')
# #     date_2 = date_2.strftime('%Y-%m-%d')
# #
# #     masks = init_masks_2(list(colors.values()), im_arr, th=7)
# #     print(f'{date_1} : start')
# #     print(f'{date_1} : mask done')
# #     board = cascaded_union(conc_list).bounds
# #
# #     product_type = 'S1_SAR_GRD'
# #     extent = {
# #         'lonmin': board[0],
# #         'lonmax': board[2],
# #         'latmin': board[1],
# #         'latmax': board[3]
# #     }
# #
# #     products, estimated_nbr_of_results = dag.search(
# #         productType=product_type,
# #         start=date_1,
# #         end=date_2,
# #         geom=extent,
# #         items_per_page=500
# #     )
