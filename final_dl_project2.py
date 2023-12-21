#get imports 
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from adabelief_pytorch import AdaBelief
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

import os
import time
import random
import warnings
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from PIL import Image
from skimage import color
from skimage.color import rgb2lab, lab2rgb



# DATASET -----------------------------------------------------------------

class ColorizationDataset(Dataset):
    def __init__(self, images, size=256, root_dir=None, transform=None, train_ratio=0.8, train=True):
        if root_dir is None:
            raise ValueError("Root directory parameter is missing in initialization of ColorizationDataset")
        else:
            self.root_dir = root_dir
        self.size = size
        self.train_ratio = train_ratio
        self.training = train
        
        # Assuming that the dataset has two folders: 'colored' and 'bw'
        if images is None:
            self.colored_images = self.get_colored_images()
            self.colored_folder = None
            self.bw_folder = None
        else:
            self.colored_images = images
            self.colored_folder = os.path.join(root_dir, 'color')
            self.bw_folder = os.path.join(root_dir, 'gray')

        # make gray images if none exist
        if images is not None and not os.path.exists(self.bw_folder):
            print("Making black and white images for", self.bw_folder)
            os.makedirs(self.bw_folder)
            self.convert2bw()

        self.transforms = transforms.Compose([
                transforms.Resize((self.size, self.size),  Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ])

    def get_colored_images(self):
        colored_images = []
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                colored_folder = os.path.join(folder_path, 'color')
                bw_folder = os.path.join(folder_path, 'gray')
                if os.path.exists(colored_folder) and os.path.exists(bw_folder):
                    colored_images.extend([os.path.join(folder, 'color', img) for img in os.listdir(colored_folder)])
        train_index = int(len(colored_images) * self.train_ratio)
        if self.training:
            return colored_images[:train_index]
        else:
            return colored_images[train_index:]

    def __len__(self):
        return len(self.colored_images)


    def __getitem__(self, index):
        # alternate loading to fix faded images
        load_path = os.path.join(self.root_dir, self.colored_images[index])
        if os.path.exists(load_path):
            colored_img_path = load_path
        else:
            colored_img_path = os.path.join(self.colored_folder, self.colored_images[index])
        
        img = Image.open(colored_img_path).convert('RGB')
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)

        L = img_lab[[0], ...] / 100. # Between 0 and 1
        ab = img_lab[[1, 2], ...] / 128. # Between -1 and 1
        return {'L': L, 'ab': ab}


    # !!! convert all images in colored_folder to grayscale and save in bw_folder
    def convert2bw(self):
        print("Converting color images to grayscale...", end=" ")
        image_types = ["jpg", "jpeg", "png"]
        rate = len(self.colored_images) // 10
        count = 0
        # get all images in self.colored_folder
        for i, file in enumerate(self.colored_images):
            if file.split(".")[-1].lower() in image_types:
                # convert to black and white, then save
                img = Image.open(os.path.join(self.colored_folder, file)).convert("L")
                img.save(self.bw_folder + file[:-4] + ".png", format='PNG')
            if i % rate == 0:
                count += 1
                print(count, end=".")
        print("done.")



# UNET GENERATOR ----------------------------------------------------------
class PenalizedTanH(nn.Module):
    def __init__(self, inplace=False):
        super(PenalizedTanH, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
          x[x > 0] = torch.tanh(x[x > 0])
          x[x <= 0] = 0.25 * torch.tanh(x[x <= 0])
          return x

        return torch.where(x > 0, torch.tanh(x), 0.25*torch.tanh(x))


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
    


# AVERAGE METER -----------------------------------------------------------
class AverageMeter:
    def __init__(self):
        self.reset()
        self.history = []

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
        self.history = []

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count
        self.history.append(self.avg)

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

# MISC UPDATE LOSS FUNCTION 
def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"\t{loss_name} = \t{loss_meter.avg:.5f}")


def plot_losses(loss_dict, idx, save=True):
    plt.subplot(1, 2, 1)
    for loss_name, loss_meter in loss_dict.items():
        plt.plot(loss_meter.history, label=loss_name)
        plt.xlabel("Batch Iteration")
        plt.ylabel("Loss")
        plt.title("All Loss Values")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss_dict["loss_G"].history)
    plt.xlabel("Batch Iteration")
    plt.ylabel("Loss")
    plt.title("Generator Loss")

    if save:
        formatted_datetime = datetime.now().strftime("%m%d%y%H%M")
        location = f"./output_images/loss_{formatted_datetime}_{idx}.png"
        print("Saving latest loss to ", location)
        plt.savefig(location)
    else:
        plt.show()



# MODEL --------------------------------------------------------------------

gen = gwenUnet()
gan_loss = GANLoss()

class Model(nn.Module):
    def __init__(self, generator=gen, loss_type="logits", adabelief=False):
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
        self.opt_G = self.set_opt(gen=True, ada=adabelief)
        self.opt_D = self.set_opt(gen=False, ada=adabelief)


    def get_data(self,data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)


    def select_loss(self):
        if self.loss_type == "logits":
            opt = nn.BCEWithLogitsLoss()
        else:
            opt = nn.BCEWithLogitsLoss()
        return opt.to(self.device)


    def set_opt(self, gen=True, lr=2e-4, betas=(0.5, 0.999), ada = False):
        # create list of parameters for the optimizer
        if ada == True:
          learning_rate = 0.001
          weight_decay = 5e-4
          eps = 1e-16  
          rectify = False 

          if gen:
              parameters = (self.net_G.parameters(), learning_rate, betas, eps, weight_decay, rectify)
          else:
              parameters = (self.net_D.parameters(), learning_rate, betas, eps, weight_decay, rectify)

          return AdaBelief(*parameters)

        else:
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
        self.loss_D_fake = self.GANLoss(self.fake_pred, target_is_real = False)

        # compute discriminator loss for real image
        real_img = torch.cat([self.L, self.ab], dim=1)
        real_pred = self.net_D(real_img)
        self.loss_D_real = self.GANLoss(real_pred, target_is_real = True)

        # discriminator backward pass
        self.loss_D = (self.loss_D_fake + self.loss_D_real) / 2
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
        # print(self.opt_D.param_groups)
        self.opt_D.step()

        # train generator
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()



#MISC FUNCTIONS - VISUALIZING FOR TRAINING ---------------------------------
def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L ) * 100.
    ab = ab * 128.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def visualize(model, data, idx, title="", save=True):
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
    if save:
        # Format the date and time as MMDDYYHHMM
        formatted_datetime = datetime.now().strftime("%m%d%y%H%M")
        location = f"./output_images/color_{formatted_datetime}_{idx}_{title}.png"
        print("Saving latest visualization to ", location)
        fig.savefig(location)


def get_last_checkpoint(path, check_name):
    # validate path parameter
    if path is None or len(path) == 0:
        return path
    if path[-1] == "/":
        path = path[:-1]
    elif path[-4:] == ".pth":
        return path

    # get all checkpoints
    valid_files = [_file for _file in os.listdir(path) if _file.endswith(".pth") and _file.startswith(check_name)]

    if not valid_files:
        return None
    else:
        # find the most recent checkpoint and return its path
        try:
            highest = max([int(file_name[:-4].split(check_name)[-1]) for file_name in valid_files])
            # mark = "" if check_name[-1] == "_" else "_"
            mark = ""
            return os.path.join(path, f"{check_name}{mark}{highest}.pth")
        except ValueError:
            print("Error: Unable to extract a valid number from file names.")
            return None


def show_time(i, idx):
    curr_time = time.time() - start
    if curr_time < 60:
        unit = "sec"
        ratio = 1
    elif curr_time < 3600:
        unit = "min"
        ratio = 60
    else:
        unit = "hrs"
        ratio = 3600
    clock = datetime.now().strftime("%H:%M")
    print(i, idx, round(curr_time / ratio, 2), unit, clock)
        


# TRAINING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# filter warnings bc we're silly geese
warnings.filterwarnings("ignore", category=UserWarning, message="Color data out of range")
warnings.filterwarnings("ignore", category=UserWarning, message="Conversion from CIE-LAB")
warnings.filterwarnings("ignore", category=UserWarning, message="CUDA initialization: The NVIDIA driver on your system is too old")

use_ada = False
use_ptanh = False
model = Model(adabelief=use_ada)
# define the root path
root_dir='./compute'
train_ratio = 0.8

# get path to the colors
print("Loading data ...", end="")
# colored_folder = os.path.join(root_dir, 'color')
# bw_folder = os.path.join(root_dir, 'gray')

# # get colored images and split into train and test
# colored_images = os.listdir(colored_folder)
# num_images_to_select = int(len(colored_images) * train_ratio)
# colored_images_train = random.sample(colored_images, num_images_to_select)
# colored_images_test = list(set(colored_images) - set(colored_images_train))

# get the train dataloader
"""note 12/11/23 the changes here were made so that the dataset class can load images from multiple
datasets aka folders. In order to do so, the root directory must be compute, which contains
each of these folders. To undo this change, simply uncomment the lines above and change the
root directory to the specific folder you want to train on"""
nw = 4
dataset_t = ColorizationDataset(images=None, root_dir=root_dir, train_ratio=train_ratio, train=True)
train_data = DataLoader(dataset_t, batch_size=16, num_workers=nw, pin_memory=True)
colored_images_train = dataset_t.colored_images

# get the validation dataloader
dataset_v = ColorizationDataset(images=None, root_dir=root_dir, train_ratio=train_ratio, train=False)
val = DataLoader(dataset_v, batch_size=16, num_workers=nw, pin_memory=True)
colored_images_test = dataset_v.colored_images
print("done.")

# def train(model, loader, epochs=35):

# #   loop = tqdm(loader, leave = True)
#   loss_dict = create_loss_meters()

#   for i in range(epochs):

#     for idx, (data_) in enumerate(loader):

#       model.get_data(data_)
#       model.optimize()
#       update_losses(model, loss_dict, count=data_['L'].size(0)) #why is this the count

#       if idx % 200 == 0:
#         log_results(loss_dict)


#     if i % 2 == 0:
#       print("\Epoch", i)
#       visualize(model, data_, i + idx)

#   return



def apply_lottery_ticket_init(model, pruning_percentage=0.2):
    # Clone the model for later use
    cloned_model = Model()
    cloned_model.load_state_dict(model.state_dict())

    # Define the pruning method (e.g., based on weights)
    def prune_weights(module, pruning_percentage):
        if isinstance(module, nn.Linear):
            weight = module.weight.data.abs().clone()
            threshold = torch.topk(weight.view(-1), int(pruning_percentage * weight.numel()), largest=False).values.max()
            mask = weight.gt(threshold).float().cuda()
            module.weight.data.mul_(mask)

    # Apply pruning to the model
    cloned_model.apply(lambda module: prune_weights(module, pruning_percentage))

    return cloned_model




def train_prune(model, loader, epochs=20, prune=False, pretrained=False,
                pretrained_path=None, check_path="./checkpoints", check_name="checkpoint",
                spacing=100, prune_wait=150):

    # load data from checkpoints

    if pretrained and pretrained_path is not None and os.path.exists(pretrained_path):
        def load_pretrained(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.opt_G.load_state_dict(checkpoint['optimizer_state_dict'][0])
            model.opt_D.load_state_dict(checkpoint['optimizer_state_dict'][1])
            return model
        print("Loading pretrained model from '", pretrained_path, "'")
        model = load_pretrained(pretrained_path)
        # loop = tqdm(loader, leave = True)
        loop = loader
        loss_dict = create_loss_meters()

    else:
        # loop = tqdm(loader, leave = True)
        loop = loader
        loss_dict = create_loss_meters()
        print("Training model from scratch")

    print("There are", len(loop), "batches per epoch.")

    # get the checkpoint number and increment
    if pretrained is not None and pretrained_path is not None and os.path.exists(pretrained_path):
        try:
            check_idx = int(pretrained_path.split(".")[-2][-1]) + 1
        except IndexError as ex:
            print(ex)
            print(pretrained_path)
    else:
        check_idx = 0


    for i in range(epochs):
        for idx, (data_) in enumerate(loop):

          # train
          show_time(i, idx)

          model.get_data(data_)
          model.optimize()
          update_losses(model, loss_dict, count=data_['L'].size(0)) #why is this the count
          #  gray = model.get_results()
          # if idx == 0:
          #   print("First iteration of epoch", i)
          #   visualize(model, data_, idx + i)

          # save new visualization
          if idx % spacing == 0:
              visualize(model, data_, idx + i, check_name)
              log_results(loss_dict)

          # prune weights
          if idx % prune_wait == 0:
              if prune == True:
                  model = apply_lottery_ticket_init(model, pruning_percentage=0.2)
                  print('PRUNED')

          # save checkpoint
          if idx % spacing == 0:
              checkpoint = {
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': [model.opt_G.state_dict(), model.opt_D.state_dict()],
                  "loss": loss_dict,
                  "iter_num": idx,
                  "epoch": i
                  # Add other relevant information like loss, accuracy, etc.
              }
              save_path = os.path.join(check_path, check_name+str(check_idx)+'.pth')
              torch.save(checkpoint, save_path)
              print("checkpoint saved to", save_path)
              check_idx += 1

          plot_losses(loss_dict, i, save=False)

    return apply_lottery_ticket_init(model, pruning_percentage=0.2)


# initializing training info
start = time.time()
to_prune = True
num_epochs = 3
checkpoint_path = "./checkpoints"
check_name = "checkpoint"
pretrained_path = get_last_checkpoint(checkpoint_path, check_name)

# show training info
print("Training started on", datetime.now().strftime("%m/%d/%y %H:%M"))
print("Using AdaBelief:", use_ada)
print("Using Penalized Tanh:", use_ptanh)
print("Pruning:", to_prune)
print("Number of epochs:", num_epochs)
print("Losses print at the end of each epoch")

train_prune(model,
            train_data,
            epochs=num_epochs,
            prune=to_prune,
            pretrained=True,
            pretrained_path=pretrained_path,
            check_path=checkpoint_path,
            check_name=check_name,
            spacing=500,
            prune_wait=500)

# def train_prune(model, loader, epochs=20, prune=False, pretrained=False, pretrained_path=None):

#   # load data from checkpoints
#   if pretrained == True and pretrained_path is not None and os.path.exists(pretrained_path):
#     def load_pretrained(path):
#       checkpoint = torch.load(path)
#       model.load_state_dict(checkpoint['model_state_dict'])
#       model.opt_G.load_state_dict(checkpoint['optimizer_state_dict'][0])
#       model.opt_D.load_state_dict(checkpoint['optimizer_state_dict'][1])
#       return model
#     print("Loading pretrained model from '", pretrained_path, "'")
#     model = load_pretrained(pretrained_path)
#     # loop = tqdm(loader, leave = True)
#     loop = loader
#     loss_dict = create_loss_meters()

#   else:
#     # loop = tqdm(loader, leave = True)
#     loop = loader
#     loss_dict = create_loss_meters()

#   print("There are", len(loop), "iterations per epoch.")

#   # get the checkpoint number and increment
#   if pretrained is not None and os.path.exists(pretrained_path):
#     check_idx = int(pretrained_path.split(".")[-2][-1]) + 1
#   else:
#     check_idx = 0

#   spacing = 100
#   prune_wait = 150
#   for i in range(epochs):
#     for idx, (data_) in enumerate(loop):

#       # train
#       curr_time = time.time() - start
#       if curr_time < 60:
#         unit = "sec"
#         ratio = 1
#       elif curr_time < 3600:
#         unit = "min"
#         ratio = 60
#       else:
#         unit = "hrs"
#         ratio = 3600  
#       clock = datetime.now().time().strftime("%H:%M")
#       print(i, idx, round(curr_time / ratio, 2), unit, clock)
      
#       model.get_data(data_)
#       model.optimize()
#       update_losses(model, loss_dict, count=data_['L'].size(0)) #why is this the count
#       gray = model.get_results()
#       if idx == 0:
#         print("First iteration of epoch", i)
#         visualize(model, data_, idx + i)

#       # save new visualization
#       if idx % spacing == 0:
#         visualize(model, data_, idx + i)
#         log_results(loss_dict)

#       # prune weights
#       if idx % prune_wait == 0:        
#         if prune == True:
#           model = apply_lottery_ticket_init(model, pruning_percentage=0.2)
#           print('PRUNED')

#       # save checkpoint
#       if idx % spacing == 0:
#         checkpoint = {
#         'epoch': i + 1,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': [model.opt_G.state_dict(), model.opt_D.state_dict()],
#         "loss": loss_dict,
#         "iter_num": idx,
#         "epoch": i
#         # Add other relevant information like loss, accuracy, etc.
#         }
#         torch.save(checkpoint, './checkpoints/checkpoint'+str(check_idx)+'.pth')
#         print("checkpoint saved to", check_idx)
#         check_idx += 1

#   return apply_lottery_ticket_init(model, pruning_percentage=0.2)


# # train(model, train_data)
# start = time.time()
# train_prune(model, 
#             train_data, 
#             prune=True, 
#             pretrained=True, 
#             pretrained_path=get_last_checkpoint("./checkpoints"))
