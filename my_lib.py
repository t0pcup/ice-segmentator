import albumentations
import numpy as np
import torch
import torchvision.ops
import segmentation_models_pytorch as smp
from torch.nn import functional
import rasterio.warp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fin_res = 128  # 128
transforms_val = albumentations.Compose([albumentations.CenterCrop(fin_res, fin_res, p=1.0, always_apply=False)])
transforms_resize_img = albumentations.Compose([albumentations.Resize(fin_res, fin_res, p=1.0, interpolation=3)])
transforms_resize_lbl = albumentations.Compose([albumentations.Resize(fin_res, fin_res, p=1.0, interpolation=0)])

model_ft = smp.DeepLabV3(
    encoder_name="timm-mobilenetv3_small_minimal_100",  # efficientnet-b0
    encoder_weights=None,
    in_channels=4,
    classes=8,
).to(device)

classes = ['other', '<1', '1-3', '3-5', '5-7', '7-9', '9-10', 'fast_ice']  # other = undefined / land / no data
palette0 = np.array([[0, 0, 0],  # other
                     [32, 32, 255],  # <1
                     [64, 64, 255],  # 1-3
                     [128, 128, 255],  # 3-5
                     [255, 255, 128],  # 5-7
                     [255, 128, 64],  # 7-9
                     [255, 64, 64],  # 9-10
                     [255, 255, 255]])  # fast_ice


def save_tiff(full_name, im, profile):
    with rasterio.open(full_name, 'w', **profile) as src:
        src.write(im)


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


def item_getter(path: str, file_name: str, transforms=transforms_val) -> (np.ndarray, np.ndarray):
    image = normalize(np.load(f'{path}/data/{file_name}'))
    label = np.load(f'{path}/label/{file_name}')

    img = image.transpose((1, 2, 0))
    augmented = transforms(image=img, mask=label[:, :])
    image = transforms_resize_img(image=augmented['image'], mask=augmented['mask'])['image']
    label = transforms_resize_lbl(image=augmented['image'], mask=augmented['mask'])['mask']
    image = image.transpose(2, 0, 1)

    assert not np.any(np.isnan(image))
    assert not np.any(np.isnan(label))
    return image, label


def tversky_loss(true, logits, alpha=0.5, beta=0.5, eps=1e-7):
    """Computes the Tversky loss [1].
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coefficient
        alpha = beta = 1 => tanimoto coefficient
        alpha + beta = 1 => F beta coefficient
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    true_1_hot = torch.eye(num_classes, device=true.device)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = functional.softmax(logits, dim=1)

    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true_1_hot.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denominate = intersection + (alpha * fps) + (beta * fns)
    return 1 - (num / (denominate + eps)).mean()
