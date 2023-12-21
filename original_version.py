#get imports 
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage import color
from skimage.color import rgb2lab, lab2rgb



# DATASET 

class ColorizationDataset(Dataset):
    def __init__(self, images, size=256, root_dir='landscape_Images', transform=None):
        self.root_dir = root_dir
        self.size = size

        # Assuming that the dataset has two folders: 'colored' and 'bw'
        self.colored_folder = os.path.join(root_dir, 'color')
        self.bw_folder = os.path.join(root_dir, 'gray')

        # make gray images if none exist
        if not os.path.exists(self.bw_folder):
            os.makedirs(self.bw_folder)
            self.convert_path2bw()

        self.transforms = transforms.Compose([
                transforms.Resize((self.size, self.size),  Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ])

        self.colored_images = images


    def __len__(self):
        return len(self.colored_images)


    def __getitem__(self, index):

        # Load colored image
        colored_img_path = os.path.join(self.colored_folder, self.colored_images[index])
        colored_img = Image.open(colored_img_path).convert('RGB')
        colored_img = self.transforms(colored_img)

        # Convert RGB image to LAB color space
        colored_lab = color.rgb2lab(colored_img)

        # Extract L, a, and b channels and normalize L to the range [0, 1]
        l_channel = colored_lab[:, :, 0] / 100.0
        a_channel = colored_lab[:, :, 1] / 128.0
        b_channel = colored_lab[:, :, 2] / 128.0

        # Convert to PyTorch tensors
        l_channel = torch.from_numpy(l_channel).float().unsqueeze(0)  # Add batch dimension
        ab_channels = torch.stack([torch.from_numpy(a_channel).float(),
                                  torch.from_numpy(b_channel).float()], dim=0)

        return {'L': l_channel, 'ab': ab_channels}


    # !!! convert all images in colored_folder to grayscale and save in bw_folder
    def convert2bw(self):
        image_types = ["jpg", "jpeg", "png"]
        # get all images in self.colored_folder
        for file in self.colored_images:
            if file.split(".")[-1].lower() in image_types:
                # convert to black and white, then save
                img = Image.open(os.path.join(self.colored_folder, file)).convert("L")
                img.save(self.bw_folder + file[:-4] + ".png")


# UNET GENERATOR 

class gwenUnet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False))
        self.layer2 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.layer3 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.layer4 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.layer5 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.layer6 = self.layer5
        self.layer7 = self.layer5
        self.layer8 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.layer9 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.layer10 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.layer11 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.layer12 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.layer13 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.layer14 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.layer15 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)

        x9 = self.layer9(torch.cat([x8, x7], dim=1))
        x10 = self.layer10(torch.cat([x9, x6], dim=1))
        x11 = self.layer11(torch.cat([x10, x5], dim=1))
        x12 = self.layer12(torch.cat([x11, x4], dim=1))
        x13 = self.layer13(torch.cat([x12, x3], dim=1))
        x14 = self.layer14(torch.cat([x13, x2], dim=1))
        x15 = self.layer15(torch.cat([x14, x1], dim=1))

        return x15
    

# DISCRIMINATOR 

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        sl = 0.02
        mom = 0.1
        eps = 1e-05

        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),                         # Input layer
                              nn.LeakyReLU(negative_slope=sl, inplace = True)                                     # Output layer
                              )
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),           # Input layer
                              nn.BatchNorm2d(128, eps=eps, momentum=mom, affine=True, track_running_stats=True),  # Activation function
                              nn.LeakyReLU(negative_slope=sl, inplace=True)                                       # Output layer
                              )
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),          # Input layer
                              nn.BatchNorm2d(256, eps=eps, momentum=mom, affine=True, track_running_stats=True),  # Activation function
                              nn.LeakyReLU(negative_slope=sl, inplace=True)                                       # Output layer
                              )
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),          # Input layer
                              nn.BatchNorm2d(512, eps=eps, momentum=mom, affine=True, track_running_stats=True),  # Activation function
                              nn.LeakyReLU(negative_slope=sl, inplace=True)                                       # Output layer
                              )
        self.layer5 = nn.Sequential(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))


    def forward(self,input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x
    

# GAN LOSS

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss
    
# MISC UPDATE LOSS FUNCTION 

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

gen = gwenUnet()
gan_loss = GANLoss() 

class Model(nn.Module):
    def __init__(self, generator = gen, loss_type="logits", up_sample=False):
        # initialization
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # learning models
        self.net_G = generator.to(self.device)
        self.net_D = Discriminator().to(self.device)

        # loss
        self.loss_type = loss_type
        self.gradient_method = "adam"
        self.loss = self.select_loss()
        self.L1 = nn.L1Loss()
        self.GANLoss = gan_loss

        # optimizers
        self.opt_G = self.set_opt(gen=True)
        self.opt_D = self.set_opt(gen=False)


    def get_data(self,data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)


    def select_loss(self):
        if self.loss_type == "logits":
            opt = nn.BCEWithLogitsLoss()
        else:
            opt = nn.BCEWithLogitsLoss()
        return opt.to(self.device)


    def set_opt(self, gen=True, lr=2e-4, betas=(0.5, 0.999)):
        # create list of parameters for the optimizer
        if gen:
            parameters = (self.net_G.parameters(), lr, betas)
        else:
            parameters = (self.net_D.parameters(), lr, betas)

        # return the desired optimizer
        if self.gradient_method == "adam":
            return optim.Adam(*parameters)
        else:
            return optim.Adam(*parameters)


    def set_requires_grad(self, model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad


    def forward(self):
        self.fake_colored = self.net_G(self.L)


    def backward_D(self):
        # compute discriminator loss for fake image
        self.fake_img = torch.cat((self.L, self.fake_colored), dim=1)
        self.fake_pred = self.net_D(self.fake_img.detach())
        # !!! we need to implement the GAN Loss
        self.loss_D_fake = self.GANLoss(self.fake_pred, target_is_real = False)

        # compute discriminator loss for real image
        real_img = torch.cat([self.L, self.ab], dim=1)
        real_pred = self.net_D(real_img)
        self.loss_D_real = self.GANLoss(real_pred, target_is_real = True)

        # discriminator backward pass
        self.loss_D = (self.loss_D_fake + self.loss_D_real) / 2
        # ??? is loss_D not a float? how are we calling backward- I think this is a tensor
        # also, I think we should divide the real loss by 2 to slow the discriminator learning
        self.loss_D.backward()

    def get_results(self):
        return self.fake_pred


    def backward_G(self):
        # get prediction from discriminator
        self.fake_img = torch.cat((self.L, self.fake_colored), dim = 1)
        fake_pred = self.net_D(self.fake_img.detach())

        # calculate loss from discriminator and normalization
        # !!! we need to implement the GAN Loss
        self.loss_G_gan = self.GANLoss(fake_pred, target_is_real = False)
        # !!! we need to implement the L1 Loss
        self.loss_G_L1 = self.L1(self.fake_colored, self.ab)
        self.loss_G = self.loss_G_gan + self.loss_G_L1

        # backward pass
        self.loss_G.backward()


    def optimize(self):
        # forward pass through generator
        self.forward()

        # train discriminator
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        # train generator
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()


# AVERAGE METER

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


# MISC FUNCTIONS FOR LOSS 
def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_gan': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

#MISC FUNCTIONS  - VISUALIZING FOR TRAINING 

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def visualize(model, data, idx, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.get_data(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_colored.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{idx}.png")



model = Model()
# define the root path
root_dir='landscape_Images'
train_ratio = 0.7

# get path to the colors
colored_folder = os.path.join(root_dir, 'color')
bw_folder = os.path.join(root_dir, 'gray')

# get colored images and split into train and test
colored_images = os.listdir(colored_folder)
num_images_to_select = int(len(colored_images) * train_ratio)
colored_images_train = random.sample(colored_images, num_images_to_select)
colored_images_test = list(set(colored_images) - set(colored_images_train))

# get the train dataloader
dataset_t = ColorizationDataset(colored_images_train)
train_data = DataLoader(dataset_t, batch_size= 16, num_workers=4, pin_memory=True)

# get the validation dataloader
dataset_v = ColorizationDataset(colored_images_test)
val = DataLoader(dataset_v, batch_size= 16, num_workers=4, pin_memory=True)


def train(model, loader, epochs = 100):

  loop = tqdm(loader, leave = True)
  loss_dict = create_loss_meters()

  for i in range(epochs):

    for idx, (data_) in enumerate(loop):

      model.get_data(data_)
      model.optimize()
      update_losses(model, loss_dict, count=data_['L'].size(0)) #why is this the count


    if i % 2 == 0:
      visualize(model, data_, i)

  return


train(model, train_data)