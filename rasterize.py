import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.plot import show
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import fiona

translate_classes = {
    '55': 1,   # ice free
    '01': 5,   # open water
    '02': 5,   # bergy water

    '10': 10,  # 1/10
    '12': 10,  # 1-2/10
    '13': 10,  # 1-3/10

    '20': 20,  # 2/10
    '23': 20,  # 2-3/10
    '24': 20,  # 2-4/10

    '30': 30,  # 3/10
    '34': 30,  # 3-4/10
    '35': 30,  # 3-5/10

    '40': 40,  # 4/10
    '45': 40,  # 4-5/10
    '46': 40,  # 4-6/10

    '50': 50,  # 5/10
    '56': 50,  # 5-6/10
    '57': 50,  # 5-7/10

    '60': 60,  # 6/10
    '67': 60,  # 6-7/10
    '68': 60,  # 6-8/10

    '70': 70,  # 7/10
    '78': 70,  # 7-8/10
    '79': 70,  # 7-9/10

    '80': 80,  # 8/10
    '89': 80,  # 8-9/10
    '81': 80,  # 8-10/10

    '90': 90,  # 9/10
    '91': 90,  # 9-10/10
    '92': 90,  # 10/10
}


def def_num(it: dict) -> int:
    # print(it['POLY_TYPE'], it['CT'])
    if it['POLY_TYPE'] == 'L':
        return 0
    # print(it['CT'])
    return translate_classes[it['CT']]


# Read in vector
file = fiona.open("E:/files/regions/2021/cis_SGRDRWA_20210614T1800Z_pl_a.shp")
# print(file.schema)


vector = gpd.read_file("E:/files/regions/2021/cis_SGRDRWA_20210614T1800Z_pl_a.shp").to_crs('epsg:4326')
# print(vector)

# Get list of geometries for all features in vector file
geom = unary_union([shapes for shapes in vector.geometry])
# print(geom)

# Open example raster
raster = rasterio.open("E:/files/view/image/2021-06-14_0_2.tiff")
# create a numeric unique value for each row
# vector['id'] = range(0, len(vector))
# print(vector)

geom_value = []
for i in range(len(file)):
    prop = file[i]
    # print(item['properties'])
    # geom_ = next(iter(vector.geometry))
    geom_ = vector.geometry[i]
    geom_value.append((geom_, def_num(prop['properties'])))

# print(geom_value)

# # create tuples of geometry, value pairs, where value is the attribute value you want to burn
# geom_value = ((geom, value) for geom, value in zip(vector.geometry, vector['id']))

# Rasterize vector using the shape and transform of the raster
rasterized = features.rasterize(geom_value,
                                out_shape=raster.shape,
                                transform=raster.transform,
                                all_touched=True,
                                fill=99,  # background value "Undetermined/Unknown"
                                merge_alg=MergeAlg.replace,
                                dtype=np.int16)
print(rasterized.shape)
# # Plot raster
# fig, ax = plt.subplots(1, figsize=(10, 10))
# show(rasterized, ax=ax)
# plt.gca().invert_yaxis()
# plt.show()
print(raster.transform)
print(raster.width, raster.height)
with rasterio.open(
        "E:/r2v.tif", "w",
        driver="GTiff",
        transform=raster.transform,
        dtype=rasterio.uint8,
        count=1,
        width=raster.width,
        height=raster.height) as dst:
    dst.write(rasterized, indexes=1)
