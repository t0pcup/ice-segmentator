import tarfile
import os


reg_path = 'E:/dafuck/regions/2021'

"""tar unpacking"""
for region in os.listdir(reg_path):
    print(region)
    for file in os.listdir(f'{reg_path}/{region}'):
        tar = tarfile.open(f'{reg_path}/{region}/{file}', "r:")
        tar.extractall(path=reg_path)
        tar.close()
