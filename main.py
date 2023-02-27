import os
import time
import copy
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score TODO
from tqdm import tqdm
from segmentation_models_pytorch.metrics import get_stats, iou_score, accuracy, balanced_accuracy, f1_score
import torch.nn as nn
import warnings
from my_lib import *

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "maxsplit_size_mb:512"
# print(os.environ["PYTORCH_CUDA_ALLOC_CONF"])


def train_model(model, loader, criterion_, optimizer,
                model_path, mod_name, dataset_size, epochs=25, save=False):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_miou = 0.0
    best_epoch = 0
    writer_dir = 'E:/files/history'
    os.makedirs(f"{writer_dir}/{v}", exist_ok=True)
    writer = SummaryWriter(f"{writer_dir}/{v}")

    for epo in range(epochs):
        torch.cuda.empty_cache()
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_pix_f1 = 0.0
            running_pix_b_acc = 0.0
            running_pix_acc = 0.0
            running_miou = 0.0
            # running_roc = 0.0
            # Iterate over data.
            for i, (inputs, labels) in enumerate(tqdm(loader[phase], desc=f'Epoch {epo}/{epochs - 1} {phase}')):
                inputs = inputs.to(device)
                labels = labels.to(device)

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
                    # loss = criterion_(labels.unsqueeze(1), outputs)  # use for dice, tversky and jaccard losses
                    loss = criterion_(outputs, labels)

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

                # TODO: get rid of l_ and p, go for labels and y_pred
                output, target = torch.LongTensor(p.astype(int)), torch.LongTensor(l_.astype(int))
                st = get_stats(output, target, mode='multiclass', num_classes=8)

                # choose reduction from: micro, macro, weighted, micro-imagewise, macro-imagewise
                red = 'weighted'
                ws = [0, 0.4, *([1] * 6)] if red == 'weighted' else None
                running_miou += iou_score(*st, reduction=red, class_weights=ws) * inputs.size(0)
                running_pix_b_acc += balanced_accuracy(*st, reduction=red, class_weights=ws) * inputs.size(0)
                running_pix_acc += accuracy(*st, reduction=red, class_weights=ws) * inputs.size(0)
                running_pix_f1 += f1_score(*st, reduction=red, class_weights=ws) * inputs.size(0)
                # print(f"micro={mi}\tmacro={ma}\tweighted={wg}\tmicro-imagewise={mii}\tmacro-imagewise={mai}\t")

            epo_loss = float(running_loss) / dataset_size[phase]
            epo_pix_f1 = float(running_pix_f1) / dataset_size[phase]
            epo_pix_b_acc = float(running_pix_b_acc) / dataset_size[phase]
            epo_pix_acc = float(running_pix_acc) / dataset_size[phase]
            epo_miou = float(running_miou) / dataset_size[phase]

            writer.add_scalar(phase + ' | loss', epo_loss, epo)
            writer.add_scalar(phase + ' | f1', epo_pix_f1, epo)
            writer.add_scalar(phase + ' | accuracy', epo_pix_acc, epo)
            writer.add_scalar(phase + ' | b_accuracy', epo_pix_b_acc, epo)
            writer.add_scalar(phase + ' | mean_IoU', epo_miou, epo)

            ph = 'test ' if phase == 'test' else 'train'
            print(f'{ph} Loss: {epo_loss:.4f} F1: {epo_pix_f1:.4f} Acc: {epo_pix_acc:.4f} ' +
                  f'bAcc: {epo_pix_b_acc:.4f} IoU: {epo_miou:.4f}')

            if phase == 'train' and save:
                torch.save({
                    'model_state_dict': model.state_dict(),
                }, f'{model_path}/{mod_name}_{epo}.pth')

            # deep copy the model
            if phase == 'test' and epo_miou > best_miou:
                best_miou = epo_miou
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epo

    time_elapsed = time.time() - since
    writer.close()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test iou: {:4f}'.format(best_miou))
    print(f'Best epoch: {best_epoch}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_miou


class CrossValDataSet(Dataset):
    def __init__(self, path, transforms_, parts, val_part, part, rgb_shift):
        if val_part > parts - 1 or val_part < 0:
            print('val_part should be < parts-1 and >=0')
        self.part = part
        self.path = path
        self.train_list = os.listdir(f'{path}/label')
        self.val_list = []

        for i in range(len(self.train_list) // parts):
            self.val_list.append(self.train_list[len(self.train_list) // parts * val_part + i])
            self.train_list[len(self.train_list) // parts * val_part + i] = 0
        while 0 in self.train_list:
            self.train_list.remove(0)
        self.rgb_shift = rgb_shift
        self.transforms = albumentations.Compose(transforms_)
        # if transforms_:
        #     self.transforms = albumentations.Compose(transforms_)
        #     self.transforms_test = albumentations.Compose(transforms_test)

    def __len__(self):
        return len([self.train_list, self.val_list][self.part == 'val'])

    def __getitem__(self, index):
        file_name = self.train_list[index] if self.part == 'train' else self.val_list[index]
        image, label = item_getter(self.path, file_name, transforms=self.transforms)
        return image, label


transforms = [
    albumentations.RandomCrop(320, 320, always_apply=False, p=1.0),
    albumentations.RandomRotate90(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
]
transforms_test = [
    albumentations.RandomCrop(320, 320, always_apply=False, p=1.0),
    # albumentations.RandomBrightnessContrast(contrast_limit=0.1, brightness_by_max=False),
    albumentations.RandomRotate90(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
]

data_dir = r'E:/files'
path_to_save = r'E:/files/pts'
os.makedirs(path_to_save, exist_ok=True)
model_name = 'NewLbl'
batch = 64
print(device, batch)

dataset = {'train': CrossValDataSet(data_dir, transforms, 5, 1, 'train', True),
           'test': CrossValDataSet(data_dir, transforms_test, 5, 1, 'val', False)}
dataloader = {'train': DataLoader(dataset['train'], batch_size=batch, shuffle=True, num_workers=0,
                                  collate_fn=coll_fn),  # , drop_last=True
              'test': DataLoader(dataset['test'], batch_size=batch, shuffle=False, num_workers=0,
                                 collate_fn=coll_fn)}  # , drop_last=True

sizes = {x: len(dataset[x]) for x in ['train', 'test']}

# criterion = tversky_loss
criterion = nn.CrossEntropyLoss(ignore_index=0)
# criterion = nn.CrossEntropyLoss(weight=torch.Tensor([[1, 0.75, 1.25]]))
# criterion = torchvision.ops.sigmoid_focal_loss

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0005)
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.7, nesterov=True)

v = 'no_brightness_crop320_w_X0.5'
model_ft, iou = train_model(model_ft, dataloader, criterion, optimizer_ft, path_to_save, model_name, sizes, 100, True)
torch.save({'model_state_dict': model_ft.state_dict()}, f'{path_to_save}/{model_name}_{v}.pth')
# torch.save(model_ft.state_dict(), f'{path_to_save}/{model_name}_v3.pth')
