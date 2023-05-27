import os
import torchvision
import segmentation_models_pytorch as smp
from vectorize import vectorize
from torch.utils.data import Dataset, DataLoader
import sys
import torch
import rasterio
import PIL.Image
import numpy as np
import albumentations as A
# from SegmentationModel import SegmentationModel
# import psycopg2
import datetime
import requests

fin_res = 256  # 128
transforms_val = A.Compose([
    A.CenterCrop(fin_res, fin_res, p=1.0, always_apply=False),
    # albumentations.Resize(fin_res, fin_res, p=1.0, interpolation=0)
])
transforms_resize_img = A.Compose([A.Resize(fin_res, fin_res, p=1.0, interpolation=3)])
transforms_resize_lbl = A.Compose([A.Resize(fin_res, fin_res, p=1.0, interpolation=0)])


def save_t(full_name, im, prof):
    with rasterio.open(full_name, 'w', **prof) as src:
        src.write(im)


def item_getter(path: str, file_name: str, transforms=transforms_val, val=False) -> (np.ndarray, np.ndarray):
    i_ = 'val_' if val else ''
    image = np.load(f'data/image/{file_name}')
    # image = normalize(np.load(f'{path}/{i_}data10-2/{file_name}'))
    # label = np.load(f'{path}/{i_}label10-5_0/{file_name}')

    img = image.transpose((1, 2, 0))
    augmented = transforms(image=img)
    image = transforms_resize_img(image=augmented['image'])['image']
    image = image.transpose((2, 0, 1))

    assert not np.any(np.isnan(image))
    return image


class InferDataset(Dataset):
    def __init__(self, path, file_n):
        self.path = path
        self.data_list = [file_n]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        f_name = self.data_list[index]
        image = item_getter(self.path, f_name, val=False)
        return image


def coll_fn(batch_):
    ims_, labels_ = [], []
    for _, sample in enumerate(batch_):
        im = sample
        ims_.append(torch.from_numpy(im.copy()))
        # labels_.append(torch.from_numpy(lbl.copy()))
    # return torch.stack(ims_, 0).type(torch.FloatTensor), torch.stack(labels_, 0).type(torch.LongTensor)
    return torch.stack(ims_, 0).type(torch.FloatTensor)


def new_normalize(im_: np.ndarray, single_norm=False, plural=False) -> np.ndarray:
    im_ = np.nan_to_num(im_)
    mean = np.array([-16.388807, -16.38885, -30.692194, -30.692194])
    std = np.array([5.6070476, 5.6069245, 8.395209, 8.395208])
    if plural:
        mean = np.array([-14.227491, -14.227545, -27.108353, -27.108353])
        std = np.array([5.096121, 5.0959415, 8.973816, 8.973816])

    if single_norm:
        mean, std = np.zeros(im_.shape[0]), np.zeros(im_.shape[0])
        for channel in range(im_.shape[0]):
            mean[channel] = np.mean(im_[channel, :, :])
            std[channel] = np.std(im_[channel, :, :])

    norm = torchvision.transforms.Normalize(mean, std)
    return np.asarray(norm.forward(torch.from_numpy(im_)))


def stand(im_: np.ndarray, single_stand=False) -> np.ndarray:
    im_ = np.nan_to_num(im_)
    min_ = np.array([-49.44221, -49.44221, -49.679745, -49.679745])
    max_ = np.array([16.50119, 15.677849, 2.95751, 2.9114623])
    if single_stand:
        min_, max_ = np.zeros(im_.shape[0]), np.zeros(im_.shape[0])
        for channel in range(im_.shape[0]):
            min_[channel] = np.min(im_[channel, :, :])
            max_[channel] = np.max(im_[channel, :, :])

    for channel in range(im_.shape[0]):
        im_[channel] = (im_[channel] - min_[channel]) / (max_[channel] - min_[channel])
    return im_


# url = sys.argv[1].replace('jpeg', 'tiff')
# url2 = sys.argv[2]
# if url2 != "null":
#     url2 = url2.replace('jpeg', 'tiff')
# else:
#     url2 = None
# order_id = sys.argv[3]
# model_name = sys.argv[4]
url = 'https://services.sentinel-hub.com/ogc/wms/########-####-####-####-############?REQUEST=GetMap&CRS=CRS:84&BBOX' \
      '=12.44693,41.870072,12.541001,41.917096&LAYERS=VV&WIDTH=512&HEIGHT=343&FORMAT=image/jpeg&TIME=2023-02-27/2023' \
      '-03-27' \
    .replace('jpeg', 'tiff')
order_id = '348e75ec-f8d4-48ca-a825-f73103713aed'
url2 = None
model_name = 'ice'

model_path = f'models/{model_name}.pth'
input_img_path = os.listdir('data/image')[0]
output_img_path = f'data/predict/output_{model_name}.tiff'


def process(url):
    # x = requests.get(url)
    # # Save input image:
    # with open(input_img_path, 'wb') as f:
    #     f.write(x.content)

    # sat_img = rasterio.open('data/image/' + input_img_path, 'r')
    # profile = sat_img.profile
    # profile['count'] = 1
    transform = A.Compose([A.Resize(256, 256, p=1.0, interpolation=3)])

    img = np.load('data/image/' + input_img_path)
    img = np.asarray(img).transpose((1, 2, 0))
    img = transform(image=img)['image']
    img = np.transpose(img, (2, 0, 1))

    img = new_normalize(img, plural=True)
    img = stand(img, single_stand=True)
    to_pil = (img[:3].transpose((1, 2, 0)) * 255).astype(np.uint8)
    PIL.Image.fromarray(to_pil).show()

    # img = img / 255.0
    # img = torch.tensor(img)

    # model = SegmentationModel()
    model = smp.DeepLabV3(
        encoder_name="timm-mobilenetv3_small_075",
        encoder_weights=None,
        in_channels=4,
        classes=6,
    ).to('cpu', dtype=torch.float32)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict']
    model.load_state_dict(state_dict)

    dataset = InferDataset('data/image', os.listdir('data/image')[0])
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=coll_fn, shuffle=False)  # 30.01.2023 was True

    inputs = next(iter(dataloader))
    inputs = inputs.to('cpu')
    model.eval()
    # ce = nn.CrossEntropyLoss()
    outputs = model(inputs)
    y_pred = torch.argmax(outputs, dim=1).cpu()
    # profile['count'] = 3
    print(np.unique(y_pred[0]))
    palette0 = np.array([
        [0, 0, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 128, 0],
        [255, 0, 0],
        [0, 128, 0],
    ])
    pred_to_pil = palette0[1 + y_pred[0, :, :]].astype(np.uint8)
    print(pred_to_pil.shape)
    PIL.Image.fromarray(pred_to_pil).show()

    # save_t(f"data/predict/alpha_pred.tiff", palette0[1 + y_pred[0, :, :]].astype(np.uint8).transpose((2, 0, 1)),
    #        profile)  # predict
    print('done')


#     # Save output image:
#     with rasterio.open(output_img_path, 'w', **profile) as src:
#         src.write(pred_mask)
#
#     result, bbox = vectorize(url)
#
#     return result, bbox
#
#
# result, bbox = process(url)
#
# result2 = None
# if url2 is not None:
#     result2, bbox = process(url2)
#
# # ------------------------------------DATABASE-----------------------------------------------------
#
# try:
#     connection = psycopg2.connect(user="postgres",
#                                   password="3172",
#                                   host="127.0.0.1",
#                                   port="5432",
#                                   database="data")
#     cursor = connection.cursor()
#
#     q = """UPDATE orders
#                 SET status = %s, url = %s, url2 = %s, finished_at = %s, result = %s, result2 = %s, bbox = %s
#                 WHERE id = %s"""
#     record = ("true", url, url2, datetime.datetime.now(), result, result2, bbox, order_id)
#
#     cursor.execute(q, record)
#
#     connection.commit()
#     count = cursor.rowcount
#     print(count, "Record inserted successfully into mobile table")
#
# except (Exception, psycopg2.Error) as error:
#     print("Failed to insert record into mobile table", error)
#
# finally:
#     # closing database connection.
#     if connection:
#         cursor.close()
#         connection.close()
#         print("PostgreSQL connection is closed")
process(url)
