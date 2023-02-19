import numpy as np
import glob
import os

path = 'D:/data'

regions = [reg.upper() for reg in os.listdir('C:/files/regions/2021') if len(reg) == 2]
print(regions)
for npy in glob.glob(f'{path}/*.shp'):
    regions