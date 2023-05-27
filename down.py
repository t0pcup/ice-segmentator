from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torchvision import transforms

# load the image
img_path = 'C:/files/data/2021-06-14_0_3.npy'
img = np.load(img_path)
print(img.shape)

# convert PIL image to numpy array
img_np = np.array(img)

# plot the pixel values
plt.hist(img_np.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")
plt.show()

# # define custom transform function TODO 1
# transform = transforms.Compose([transforms.ToTensor()])
#
# # transform the pIL image to tensor
# # image
# img_tr = transform(img)
#
# # Convert tensor image to numpy array
# img_np = np.array(img_tr)
#
# # plot the pixel values
# plt.hist(img_np.ravel(), bins=50, density=True)
# plt.xlabel("pixel values")
# plt.ylabel("relative frequency")
# plt.title("distribution of pixels")
# plt.show()
#
# # todo 2
#
# # get tensor image
# img_tr = transform(img)
#
# # calculate mean and std
# print(img_np.shape)
# mean = [np.mean(img_np[:, m, :]) for m in range(img_np.shape[1])]
# std = [np.std(img_np[:, m, :]) for m in range(img_np.shape[1])]
mean = [np.mean(img_np[m]) for m in range(img_np.shape[0])]
std = [np.std(img_np[m]) for m in range(img_np.shape[0])]

# print mean and std
print("mean and std before normalize:")
print("Mean of the image:", mean)
print("Std of the image:", std)

# todo 3
# define custom transform
# here we are using our calculated
# mean & std
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# get normalized image
img = img.transpose((1, 2, 0))
print(img.shape)
img_normalized = transform_norm(img)
# convert normalized image to numpy
# array
img_np = np.array(img_normalized)

# plot the pixel values
plt.hist(img_np.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")
plt.show()
# get normalized image
img_normalized = transform_norm(img)

# convert this image to numpy array
img_normalized = np.array(img_normalized)

# transpose from shape of (3,,) to shape of (,,3)
img_normalized = img_normalized.transpose((1, 2, 0))

# display the normalized image
plt.imshow(img_normalized)
plt.xticks([])
plt.yticks([])
plt.show()

# get normalized image
img_nor = transform_norm(img)

# calculate mean and std
mean, std = img_nor.mean([1, 2]), img_nor.std([1, 2])

# print mean and std
print("Mean and Std of normalized image:")
print("Mean of the image:", mean)
print("Std of the image:", std)

# import os
# import boto3
# from tqdm import tqdm
#
# session = boto3.session.Session()
# s3_client = session.client(
#     service_name='s3',
#     endpoint_url='###########',
#     aws_access_key_id='###########',
#     aws_secret_access_key='###########'
# )
#
# bucket_name = 'ds-aurora'
# dir_d = 'datasets/DS-ICE.1.4/ice_water_v3/Data'
# paginator = s3_client.get_paginator('list_objects_v2')
# pages = paginator.paginate(Bucket=bucket_name, Prefix=dir_d)
#
# keys = []
# for page in pages:
#     for key in page['Contents']:
#         keys.append(key['Key'])
#         # print(key['Key'])
#
# lst = os.listdir('C:/files/label')
# for file_path in tqdm(lst, total=len(lst)):
#     file_path = f'datasets/DS-ICE.1.4/ice_water_v3/Data/{file_path}'
#     file_name = file_path.split('/')[-1]
#
#     response = s3_client.get_object(Bucket=bucket_name, Key=file_path)
#     with open(f'C:/files/data1/{file_name}', 'wb') as f:
#         f.write(response['Body'].read())
#
#     file_name = file_path.replace(".npy", ".tiff").split('/')[-1]
#     response = s3_client.get_object(Bucket=bucket_name, Key=file_path)
#     with open(f'C:/files/data1/{file_name}', 'wb') as f:
#         f.write(response['Body'].read())
