import numpy as np
import os
import time
import copy
import torch
import torchvision.ops
import torch.optim as optim
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
import albumentations
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")
#
# cuda_id = torch.cuda.current_device()
# print(f"ID of current CUDA device:{torch.cuda.current_device()}")
#
# print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

loss_history, test_loss_history = [], []
f1_history, test_f1_history = [], []
miou_history, test_miou_history = [], []
acc_history, test_acc_history = [], []


# roc_history, test_roc_history = [], [] TODO


def fin_plotter(train_h, test_h, color, title, xt_, yt_, xl_, yl_):
    plt.figure(dpi=250)
    plt.plot(train_h, color=color, label='train', alpha=0.4)
    plt.plot(test_h, color=color, label='test')
    plt.xticks(xt_)
    plt.yticks(yt_)
    plt.xlim(0, xl_)
    plt.ylim(*yl_)
    plt.grid(True)
    plt.title(title)
    plt.legend()
    plt.show()


def double_exponential_smoothing(data, alpha=0.0, beta=0.0):
    result = [data[0]]
    for n in range(1, len(data) + 1):
        if n == 1:
            level, trend = data[0], data[1] - data[0]
        if n >= len(data):  # прогнозируем
            value = result[-1]
        else:
            value = data[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)

    return [result[0:len(result) - 1], f"Double exp: a{alpha}, b{beta}"]


def clear_gpu():
    import gc
    try:
        del model_ft
    except NameError:
        print('There is no model with that name')
    gc.collect()
    torch.cuda.empty_cache()


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


def train_model(model, device_, dataset_loader, criterion_, optimizer,
                model_path, mod_name, dataset_size, epochs=25, save=False):
    def mean_iou(y_true, y_pred_):
        _and = np.logical_and(y_true, y_pred_).sum()
        _or = np.logical_or(y_true, y_pred_).sum()
        return _and / _or

    since = time.time()
    global loss_history, f1_history, miou_history, acc_history
    global test_loss_history, test_f1_history, test_miou_history, test_acc_history
    best_model_wts = copy.deepcopy(model.state_dict())
    best_miou = 0.0
    best_epoch = 0
    # ce = nn.CrossEntropyLoss(ignore_index=0)

    for epo in range(epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_pix_f1 = 0.0
            running_pix_acc = 0.0
            running_miou = 0.0
            running_roc = 0.0
            ls = np.array([])
            ps = np.array([])
            # Iterate over data.
            for i, (inputs, labels) in enumerate(tqdm(dataset_loader[phase], desc=f'Epoch {epo}/{epochs - 1} {phase}')):
                inputs = inputs.to(device_)
                labels = labels.to(device_)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = outputs.to(device)

                    # y_pred = torch.argmax(outputs['out'], dim=1)
                    y_pred = torch.argmax(outputs, dim=1)
                    #                     loss = criterion_(torch.flatten(outputs.permute(0,2,3,1), end_dim=8),
                    #                                     torch.flatten(labels))
                    # loss = criterion_(labels.unsqueeze(1), outputs['out'])
                    # loss = criterion_(outputs['out'], labels, ce)
                    # loss = criterion_(outputs, labels, ce)
                    criterion_ = criterion_.to(device)
                    loss = criterion_(outputs, labels)
                    # loss = criterion_(labels.unsqueeze(1), outputs)  # use for dice, tversky and jaccard losses

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                #                         if scheduler:
                #                             scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                labels = labels.float()

                l_ = torch.flatten(labels).cpu().detach().numpy()
                p = torch.flatten(y_pred).cpu().detach().numpy()
                if len(ls) == 0:
                    ls = l_
                    ps = p
                else:
                    np.append(ls, l_)
                    np.append(ps, p)
                running_pix_acc += balanced_accuracy_score(l_, p) * inputs.size(0)  # , sample_weight=[1, 0.9, 1]
                running_pix_f1 += f1_score(l_, p, average='micro') * inputs.size(0)
                running_miou += mean_iou(l_, p) * inputs.size(0)
                # running_roc += roc_auc_score(l_, p) * inputs.size(0)

            epoch_loss = running_loss / dataset_size[phase]
            mean_epoch_pix_f1 = running_pix_f1 / dataset_size[phase]
            epoch_pix_acc = running_pix_acc / dataset_size[phase]
            epoch_miou = running_miou / dataset_size[phase]
            epoch_roc = running_roc / dataset_size[phase]

            # print('{} Loss: {:.4f} Acc: {:.4f} IoU: {:.4f}'.format(
            #     phase, epoch_loss, epoch_pix_acc, epoch_miou))
            if phase == 'train':
                loss_history.append(epoch_loss)
                f1_history.append(mean_epoch_pix_f1)
                acc_history.append(epoch_pix_acc)
                miou_history.append(epoch_miou)
                # roc_history.append(epoch_roc) TODO
            else:
                test_loss_history.append(epoch_loss)
                test_f1_history.append(mean_epoch_pix_f1)
                test_acc_history.append(epoch_pix_acc)
                test_miou_history.append(epoch_miou)
                # test_roc_history.append(epoch_roc) TODO

            print('{} Loss: {:.4f} F1: {:.4f} Acc: {:.4} IoU: {:.4f}'.format(
                phase, epoch_loss, mean_epoch_pix_f1, epoch_pix_acc, epoch_miou))

            if epo == 0 and phase == 'train':
                min_y = min([f1_history[0], miou_history[0], acc_history[0]])
                max_y = loss_history[0]
            else:
                if phase == 'train':
                    min_y = min([min(f1_history), min(miou_history), min(acc_history)])
                    max_y = max(loss_history)
                else:
                    min_y = min([min(test_f1_history), min(test_miou_history), min(test_acc_history)])
                    max_y = max(test_loss_history)
            # min(roc_history), min(test_roc_history), TODO

            yl = [min_y - 0.1, max_y + 0.1]
            yt = np.arange(yl[0], yl[1], 0.1)
            st = 5 * (1 + int(epo > 70) + int(epo > 150))
            xt = [st * i for i in range((epo % st != 0) + 1 + epo // st)]

            # if epo + 1 == epochs and phase == 'test':
            #     plt.figure(dpi=250)
            #     plt.plot(loss_history, color='r', alpha=0.4)
            #     plt.plot(f1_history, color='g', alpha=0.4)
            #     plt.plot(acc_history, color='orange', alpha=0.4)
            #     plt.plot(miou_history, color='b', alpha=0.4)
            #     # plt.plot(roc_history, color='purple', alpha=0.4) TODO
            #     plt.plot(test_loss_history, color='r', label='loss')
            #     plt.plot(test_f1_history, color='g', label='F1')
            #     plt.plot(test_acc_history, color='orange', label='Acc')
            #     plt.plot(test_miou_history, color='b', label='IoU')
            #     # plt.plot(test_roc_history, color='purple', label='ROC') TODO
            #     plt.xticks(xt)
            #     plt.yticks(yt)
            #     plt.xlim(0, epo)
            #     plt.ylim(*yl)
            #     plt.grid(True)
            #     plt.title('Fin plots comparison')
            #     plt.legend()
            #     plt.show()
            #
            #     fin_plotter(loss_history, test_loss_history, 'r', 'Fin Loss comparison', xt, yt, epo, yl)
            #     fin_plotter(f1_history, test_f1_history, 'g', 'Fin F1 comparison', xt, yt, epo, yl)
            #     fin_plotter(miou_history, test_miou_history, 'b', 'Fin IoU comparison', xt, yt, epo, yl)
            #     fin_plotter(acc_history, test_acc_history, 'orange', 'Fin Acc comparison', xt, yt, epo, yl)
            #     # fin_plotter(roc_history, test_roc_history, 'orange', 'Fin Acc comparison', xt, yt, epo, yl) TODO

            if epo % 10 == 0 or epo + 1 == epochs:
                plt.figure(dpi=250)
                choice = phase == 'test'
                xs = [loss_history, test_loss_history][choice]
                plt.plot(xs, color='r', label='loss')
                plt.plot([np.min(xs) for _ in xs], color='r', alpha=0.4)

                xs = [miou_history, test_miou_history][choice]
                plt.plot(xs, color='b', label='IoU')
                plt.plot([np.max(xs) for _ in xs], color='b', alpha=0.4)

                xs = [f1_history, test_f1_history][choice]
                plt.plot(xs, color='g', label='F1')
                plt.plot([np.max(xs) for _ in xs], color='g', alpha=0.4)

                xs = [acc_history, test_acc_history][choice]
                plt.plot(xs, color='orange', label='Acc')
                plt.plot([np.max(xs) for _ in xs], color='orange', alpha=0.4)

                plt.xticks(xt)
                plt.yticks(yt)
                plt.xlim(0, epo)
                plt.ylim(*yl)
                plt.grid(True)
                plt.title(f'{phase} plots')
                plt.legend()
                plt.show()

            if phase == 'train' and save:
                torch.save({
                    'model_state_dict': model.state_dict(),
                }, f'{model_path}/{mod_name}_{epo}.pth')

            # deep copy the model
            if phase == 'test' and epoch_miou > best_miou:  # TODO: try accuracy
                best_miou = epoch_miou
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epo
            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test iou: {:4f}'.format(best_miou))
    print(f'Best epoch: {best_epoch}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_miou


test_iou = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
batch = 16

data_dir = r'E:/files'
classes = ['land', 'water', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'no data', 'ice shelf', 'undefined']
path_to_save = r'E:/files/pts'
os.makedirs(path_to_save, exist_ok=True)
model_name = 'NewLbl'


class CrossValDataSet(Dataset):
    def __init__(self, datapath, transforms_, parts, val_part, part, rgb_shift):
        if val_part > parts - 1 or val_part < 0:
            print('val_part should be < parts-1 and >=0')
        self.part = part
        self.datapath = datapath
        self.train_list = os.listdir(f'{datapath}/label')
        self.val_list = []

        for i in range(len(self.train_list) // parts):
            self.val_list.append(self.train_list[len(self.train_list) // parts * val_part + i])
            self.train_list[len(self.train_list) // parts * val_part + i] = 0
        while 0 in self.train_list:
            self.train_list.remove(0)
        self.rgb_shift = rgb_shift
        self.transforms = None
        if transforms_:
            self.transforms = albumentations.Compose(transforms_)
            self.transforms_test = albumentations.Compose(transforms_test)
        self.transforms_val = albumentations.Compose(transforms_val)

    def __len__(self):
        return len([self.train_list, self.val_list][self.part == 'val'])

    def __getitem__(self, index):
        file_name = self.train_list[index] if self.part == 'train' else self.val_list[index]

        img = np.load(f'{self.datapath}/data/{file_name}')
        label = np.load(f'{self.datapath}/label/{file_name}')

        # min_v, max_v = np.min(img), np.max(img)
        min_v, max_v = -84.64701, 4.379311  # TODO: сравнить
        img = ((img + min_v) / (abs(max_v) - abs(min_v) + 1) * 255).astype(np.uint8)

        img = img.transpose(1, 2, 0)
        if self.transforms:
            augmented = self.transforms_test(image=img, mask=label[:, :])
        else:
            augmented = self.transforms_val(image=img, mask=label[:, :])
        img = augmented['image']
        label = augmented['mask']

        img = img.transpose(2, 0, 1)
        return img, label


def coll_fn(batch_):
    imgs_, labels_ = [], []
    for _, sample in enumerate(batch_):
        img, label = sample
        imgs_.append(torch.from_numpy(img.copy()))
        labels_.append(torch.from_numpy(label.copy()))
    return torch.stack(imgs_, 0).type(torch.FloatTensor), torch.stack(labels_, 0).type(torch.LongTensor)


transforms = [
    albumentations.RandomBrightnessContrast(contrast_limit=0.1, brightness_by_max=False),
    albumentations.RandomBrightnessContrast(contrast_limit=0.1, brightness_by_max=False),
    albumentations.RandomRotate90(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.RandomCrop(256, 256, always_apply=False, p=1.0)
]
transforms_test = [
    albumentations.Resize(256, 256),
    albumentations.RandomBrightnessContrast(contrast_limit=0.1, brightness_by_max=False),
    albumentations.RandomRotate90(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
]
transforms_val = [albumentations.Resize(1280, 1280, interpolation=3)]

dataset = {'train': CrossValDataSet(data_dir, transforms_val, 5, 1, 'train', True),
           'test': CrossValDataSet(data_dir, transforms_val, 5, 1, 'val', False)}
dataloader = {'train': DataLoader(dataset['train'], batch_size=batch, shuffle=True, num_workers=0,
                                  collate_fn=coll_fn),  # , drop_last=True
              'test': DataLoader(dataset['test'], batch_size=batch, shuffle=False, num_workers=0,
                                 collate_fn=coll_fn)}  # , drop_last=True

dataset_sizes = {x: len(dataset[x]) for x in ['train', 'test']}

model_ft = smp.DeepLabV3(
    encoder_name="timm-mobilenetv3_small_minimal_100",  # efficientnet-b0
    encoder_weights=None,
    in_channels=4,
    classes=14,
)

model_ft = model_ft.to(device)
# criterion = tversky_loss
criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss(weight=torch.Tensor([[1, 0.75, 1.25]]))
# criterion = torchvision.ops.sigmoid_focal_loss

# optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01)
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.7, nesterov=True)

model_ft, iou = train_model(model_ft, device, dataloader, criterion, optimizer_ft, path_to_save, model_name,
                            dataset_sizes, 100, False)

torch.save({'model_state_dict': model_ft.state_dict()}, f'{path_to_save}/{model_name}_v0.1.pth')
# torch.save(model_ft.state_dict(), f'{path_to_save}/{model_name}_v3.pth')

# TODO: 1. посмотреть на карте (узнать как наложить labels/masks на карту)
# 2. f1_score на общей матрице +
# 3. пересобрать патчи (1.2 -> 2.5) +
# 4. пилить графики +
# TODO: * для участка построить карту labels и карту predicts (+img) !!!
# banned: 2021-06-14_3_1
