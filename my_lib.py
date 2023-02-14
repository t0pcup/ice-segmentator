import albumentations
import numpy as np
import torch
import torchvision.ops
import segmentation_models_pytorch as smp


def normalize(im: np.ndarray) -> np.ndarray:
    im = np.nan_to_num(im)
    mean, std = np.zeros(im.shape[0]), np.zeros(im.shape[0])
    for channel in range(im.shape[0]):
        mean[channel] = np.mean(im[channel, :, :])
        std[channel] = np.std(im[channel, :, :])

    norm = torchvision.transforms.Normalize(mean, std)
    return np.asarray(norm.forward(torch.from_numpy(im)))


def coll_fn(batch_):
    ims_, labels_ = [], []
    for _, sample in enumerate(batch_):
        im, lbl = sample
        ims_.append(torch.from_numpy(im.copy()))
        labels_.append(torch.from_numpy(lbl.copy()))
    return torch.stack(ims_, 0).type(torch.FloatTensor), torch.stack(labels_, 0).type(torch.LongTensor)


def item_getter(path: str, file_name: str, transforms) -> (np.ndarray, np.ndarray):
    image = normalize(np.load(f'{path}/data/{file_name}'))
    label = np.load(f'{path}/label/{file_name}')

    resize_img = [albumentations.Resize(128, 128, p=1.0, interpolation=3)]
    resize_lbl = [albumentations.Resize(128, 128, p=1.0, interpolation=0)]

    transforms_resize_image = albumentations.Compose(resize_img)
    transforms_resize_label = albumentations.Compose(resize_lbl)

    img = image.transpose((1, 2, 0))
    augmented = transforms(image=img, mask=label[:, :])
    image = transforms_resize_image(image=augmented['image'], mask=augmented['mask'])['image']
    label = transforms_resize_label(image=augmented['image'], mask=augmented['mask'])['mask']
    image = image.transpose(2, 0, 1)

    assert not np.any(np.isnan(image))
    assert not np.any(np.isnan(label))
    return image, label


model_ft = smp.DeepLabV3(
    encoder_name="timm-mobilenetv3_small_minimal_100",  # efficientnet-b0
    encoder_weights=None,
    in_channels=4,
    classes=8,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
