# %% [markdown]
# # One Shot Learning with Siamese Networks
#
# This is the jupyter notebook that accompanies

# %% [markdown]
# ## Imports
# All the imports are defined here
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
from helper import *

if __name__ == '__main__':

    # ## Training Time!
    #train_model()
    net = SiameseNetwork()
    net.load_state_dict(torch.load("./trained_siam_net_model.pt"))
    net.eval()

    # ## Some simple testing
    # The last 3 subjects were held out from the training, and will be used to test. The Distance between each image pair denotes the degree of similarity the model found between the two images. Less means it found more similar, while higher values indicate it found them to be dissimilar.

    face_match(net, "./target")
    #visualise_differences(net)
