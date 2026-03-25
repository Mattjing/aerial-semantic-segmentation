
from IPython.display import clear_output
!pip install segmentation-models-pytorch
clear_output()


!pip install --upgrade certifi
clear_output()


import os
os.environ['LD_LIBRARY_PATH'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\libnvvp;' + os.environ.get('LD_LIBRARY_PATH', '')
print("PATH:", os.environ['PATH'])
print("LD_LIBRARY_PATH:", os.environ['LD_LIBRARY_PATH'])



import time

from PIL import Image

import numpy as np
import pandas as pd

import cv2

import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split

import torch

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from tqdm.notebook import tqdm

import albumentations as A

import segmentation_models_pytorch as smp


IMAGE_PATH = 'C:/Users/mattj/Documents/McMaster/MEST/SEP 769/archive/dataset/semantic_drone_dataset/original_images/'
MASK_PATH = 'C:/Users/mattj/Documents/McMaster/MEST/SEP 769/archive/dataset/semantic_drone_dataset//label_images_semantic/'


print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


class_df = pd.read_csv('C:/Users/mattj/Documents/McMaster/MEST/SEP 769/archive/class_dict_seg.csv')
class_df


CLASSES = class_df.name.to_list()
CLASSES


n_classes = len(CLASSES)
n_classes


# Remove spaces in the column name column
class_df.columns = class_df.columns.str.strip()
class_df['name'] = class_df['name'].str.strip()

# Create a dictionary that maps labels to colors
class_dict = {row['name']: (row['r'], row['g'], row['b']) for _, row in class_df.iterrows()}

# Print out the color legend and corresponding label
for label, color in class_dict.items():
    print(f'Label: {label}, Color: {color}')

def label_to_color(mask, class_dict):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in class_dict.items():
        mask_label = np.where(mask == list(class_dict.keys()).index(label), 1, 0)
        color_mask[mask_label == 1] = color
    return color_mask


def create_df():
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            name.append(filename.split('.')[0])

    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

df_images = create_df()
print('Total Images: ', len(df_images))


# Split the data into train and test sets
X_trainval, X_test = train_test_split(df_images['id'].values, test_size=0.1, random_state=19)
X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)

print('Train Size   : ', len(X_train))
print('Val Size     : ', len(X_val))
print('Test Size    : ', len(X_test))


# Show a sample image with its size information
img = Image.open(IMAGE_PATH + df_images['id'][100] + '.jpg')
mask = Image.open(MASK_PATH + df_images['id'][100] + '.png')
print('Image Size', np.asarray(img).shape)
print('Mask Size', np.asarray(mask).shape)


plt.imshow(img)
plt.imshow(mask, alpha=0.6)
plt.title('Picture with Mask Appplied')
plt.show()


class DroneDataset(Dataset):

    def __init__(self, img_path, mask_path, X, mean, std, n_model,transform=None, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.patches = patch
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()

        if self.patches:
            img, mask = self.tiles(img, mask)

        return img, mask

    def tiles(self, img, mask):

        img_patches = img.unfold(1, 512, 512).unfold(2, 768, 768)
        img_patches  = img_patches.contiguous().view(3,-1, 512, 768)
        img_patches = img_patches.permute(1,0,2,3)

        mask_patches = mask.unfold(0, 512, 512).unfold(1, 768, 768)
        mask_patches = mask_patches.contiguous().view(-1, 512, 768)

        return img_patches, mask_patches


mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

# Use albumentations library for image data enhancement
t_train = A.Compose([
    A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST),  # Resize photos
    A.HorizontalFlip(),  # Flip the photo horizontally
    A.VerticalFlip(),    # Flip the photo vertically
    A.GridDistortion(p=0.2),  # Mesh deformation with 20% probability
    A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),  # Adjust brightness and contrast randomly
    A.GaussNoise()  # Add Gaussian noise
])

t_val = A.Compose([
    A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST),  # Resize photos
    A.HorizontalFlip(),  # Flip the photo horizontally
    A.GridDistortion(p=0.2)  # Mesh deformation with 20% probability
])

# Data augmentation settings for training and testing datasets
train_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std, t_train, patch=False)
val_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std, t_val, patch=False)

# Introduce DataLoader for training and testing
batch_size = 3
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=23):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, patch=False):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []; val_acc = []
    train_iou = []; train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1 ; not_improve=0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1,c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device); mask = mask_tiles.to(device);
            output = model(image)
            loss = criterion(output, mask)
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    image_tiles, mask_tiles = data

                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1,c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)

                    image = image_tiles.to(device); mask = mask_tiles.to(device);
                    output = model(image)
                    val_iou_score +=  mIoU(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    loss = criterion(output, mask)
                    test_loss += loss.item()

            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(val_loader))


            if min_loss > (test_loss/len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss/len(val_loader))))
                min_loss = (test_loss/len(val_loader))
                decrease += 1
                if decrease % 5 == 0:
                    print('saving model...')
                    torch.save(model.state_dict(), model_name+'_mIoU-{:.3f}.pt'.format(val_iou_score/len(val_loader)))

            # Stop condition when reaching mIoU 1.0
            if val_iou_score / len(val_loader) >= 1:
                print(f"Reached mIoU 0.9, stop training at epoch {e+1}.")
                break

            val_iou.append(val_iou_score/len(val_loader))
            train_iou.append(iou_score/len(train_loader))
            train_acc.append(accuracy/len(train_loader))
            val_acc.append(test_accuracy/ len(val_loader))
            print("Epoch:{}/{}..".format(e+1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss/len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss/len(val_loader)),
                  "Train mIoU:{:.3f}..".format(iou_score/len(train_loader)),
                  "Val mIoU: {:.3f}..".format(val_iou_score/len(val_loader)),
                  "Train Acc:{:.3f}..".format(accuracy/len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy/len(val_loader)),
                  "Time: {:.2f}m".format((time.time()-since)/60))

    history = {'train_loss' : train_losses, 'val_loss': test_losses,
               'train_miou' :train_iou, 'val_miou':val_iou,
               'train_acc' :train_acc, 'val_acc':val_acc,
               'lrs': lrs}
    print('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    return history


def plot_loss(history):
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot( history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch'); plt.ylabel('loss');
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

def plot_score(history):
    plt.plot(history['train_miou'], label='train_mIoU', marker='*')
    plt.plot(history['val_miou'], label='val_mIoU',  marker='*')
    plt.title('Score per epoch'); plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

def plot_acc(history):
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy',  marker='*')
    plt.title('Accuracy per epoch'); plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


class DroneTestDataset(Dataset):

    def __init__(self, img_path, mask_path, X, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            img = Image.fromarray(img)

        mask = torch.from_numpy(mask).long()

        return img, mask


t_test = A.Resize(768, 1152, interpolation=cv2.INTER_NEAREST)
test_set = DroneTestDataset(IMAGE_PATH, MASK_PATH, X_test, transform=t_test)


def predict_image_mask_miou(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score


def predict_image_mask_pixel(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc


def miou_score(model, test_set):
    score_iou = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
    return score_iou


def pixel_acc(model, test_set):
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, acc = predict_image_mask_pixel(model, img, mask)
        accuracy.append(acc)
    return accuracy


max_lr = 1e-3
epoch = 100
weight_decay = 1e-4

n_model = 'efficientnet-b3'


model_name = 'fpn_efficientnet_b3'

model = smp.FPN(
    encoder_name=n_model,
    encoder_weights="imagenet",
    in_channels=3,
    classes=n_classes,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                            steps_per_epoch=len(train_loader))

# Port model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

history = fit(epoch, model, train_loader, val_loader, criterion, optimizer, sched, model_name)

torch.save(model.state_dict(), model_name + '_save.pt')

model


mob_miou = miou_score(model, test_set)
mob_acc = pixel_acc(model, test_set)

print('Model: ', model_name)
print('Test Set mIoU', np.mean(mob_miou))
print('Test Set Pixel Accuracy', np.mean(mob_acc))


plot_acc(history)


plot_loss(history)


plot_score(history)


for i in range(10):
    image, mask = test_set[i]
    pred_mask, score = predict_image_mask_miou(model, image, mask)

    # Apply colors to predicted and actual labels
    pred_mask_color = label_to_color(pred_mask, class_dict)
    mask_color = label_to_color(mask, class_dict)

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.imshow(image)
    ax1.set_title('Ảnh gốc')
    ax1.set_axis_off()

    ax2.imshow(mask_color)
    ax2.set_title('Ground Truth')
    ax2.set_axis_off()

    ax3.imshow(pred_mask_color)
    ax3.set_title(f'{model_name} | mIoU {score:.3f}')
    ax3.set_axis_off()

    # Draw a legend for the label
    legend_elements = []
    unique_labels = np.unique(pred_mask)
    for label in unique_labels:
        label_name = list(class_dict.keys())[label]
        color = class_dict[label_name]
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=np.array(color)/255, markersize=10, label=label_name)
        )

    # Set legend in ax4, disable display of axes in ax4
    ax4.legend(handles=legend_elements, title='Nhãn')
    ax4.axis('off')

    plt.tight_layout()
    plt.show()
