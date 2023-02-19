import tarfile
import os
from tqdm import tqdm


# reg_path = 'E:/dafuck/regions/2021'
reg_path = 'C:/files/regions/2021'

"""tar unpacking"""
for region in os.listdir(reg_path):
    for file in tqdm(os.listdir(f'{reg_path}/{region}'), desc=region):
        tar = tarfile.open(f'{reg_path}/{region}/{file}', "r:")
        tar.extractall(path=reg_path)
        tar.close()
