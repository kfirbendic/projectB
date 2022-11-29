#matplotlib
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


# %% [markdown]
# ## Helper functions
# Set of helper functions

# %%
def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()

# ## Configuration Class
# A simple class to manage configuration

# %%
class Config():
    training_dir = "./data/faces/training/"
    testing_dir = "./data/faces/testing/"
    train_batch_size = 64
    train_number_epochs = 100


# ## Custom Dataset Class
# This dataset generates a pair of images. 0 for geniune pair and 1 for imposter pair

# %%
class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        print(img0_tuple[1])
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class SiameseNetworkDatasetTest(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = self.imageFolderDataset.imgs[index]
        img0 = Image.open(img0_tuple[0])
        img0 = img0.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)

        if self.transform is not None:
            img0 = self.transform(img0)

        return img0, img0_tuple[1]

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
# ## Neural Net Definition
# We will use a standard convolutional neural network

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# %%
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def train_model():
    # ## Using Image Folder Dataset
    folder_dataset = dset.ImageFolder(root=Config.training_dir)

    # %%
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor()
                                                                          ])
                                            , should_invert=False)
    print(torch.cuda.is_available())
    # ## Visualising some of the data
    # The top row and the bottom row of any column is one pair. The 0s and 1s correspond to the column of the image.
    # 1 indiciates dissimilar, and 0 indicates similar.

    vis_dataloader = DataLoader(siamese_dataset,
                                shuffle=True,
                                num_workers=8,
                                batch_size=8)
    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    imshow(torchvision.utils.make_grid(concatenated))
    print(example_batch[2].numpy())

    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=8,
                                  batch_size=Config.train_batch_size)

    net = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, Config.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0, img1, label
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

    #torch.save(net.state_dict(), "./trained_siam_net_model.pt")
    show_plot(counter, loss_history)

def visualise_differences(net):
    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor()
                                                                          ])
                                            , should_invert=False)
    test_dataloader = DataLoader(siamese_dataset, num_workers=6, batch_size=1, shuffle=False)
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)
    for i in range(10):
        _, x1, label2 = next(dataiter)
        concatenated = torch.cat((x0, x1), 0)

        output1, output2 = net(Variable(x0), Variable(x1))
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))

def face_match(net, target_dir):
    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
    siamese_dataset_test = SiameseNetworkDatasetTest(imageFolderDataset=folder_dataset_test,
                                                     transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                                   transforms.ToTensor()
                                                                                   ])
                                                     , should_invert=False)
    loader_test = DataLoader(siamese_dataset_test, num_workers=1, batch_size=1, shuffle=False)

    target_image_dataset = dset.ImageFolder(root=target_dir)
    siamese_dataset_target = SiameseNetworkDatasetTest(imageFolderDataset=target_image_dataset,
                                                       transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                                     transforms.ToTensor()
                                                                                     ])
                                                       , should_invert=False)
    loader_target = DataLoader(siamese_dataset_target, num_workers=1, batch_size=1, shuffle=False)

    idx_to_class = {i: c for c, i in
                    folder_dataset_test.class_to_idx.items()}  # accessing names of peoples from folder names
    dict_of_dist = {}

    dataiter = iter(loader_target)
    img_test = next(dataiter)[0]

    for img, idx in loader_test:
        output1, output2 = net(Variable(img_test), Variable(img))
        euclidean_distance = F.pairwise_distance(output1, output2)
        if idx[0].item() in dict_of_dist:
            dict_of_dist[idx[0].item()].append(euclidean_distance.item())
        else:
            dict_of_dist[idx[0].item()] = [euclidean_distance.item()]

    for i in dict_of_dist:
        dict_of_dist[i] = sum(dict_of_dist[i]) / len(dict_of_dist[i])

    print(idx_to_class[(min(dict_of_dist, key=dict_of_dist.get))])