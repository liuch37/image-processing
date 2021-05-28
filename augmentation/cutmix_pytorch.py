'''
This code is to implement cutmix method proposed in
https://arxiv.org/abs/1905.04899
as its image augmentation method, using PyTorch.
'''

import torch
import torchvision.transforms as transforms
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import pdb

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(input, target, beta=1.0):
    '''
    input: 
    input: batch of tensor RGB images [batch, 3, H, W]
    target: batch of tensor label [batch, class]
    beta: hyperparameter for beta distribution

    output: 
    a cutmix image [batch, 3, H, W]
    '''
    # generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(img.size()[0])
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

    return input, target_a, target_b, lam

# unit testing
if __name__ == '__main__':
    # input path
    test_image1_path = '../test_images/0009.jpg'
    test_image2_path = '../test_images/0029.jpg'

    # image I/O and processing
    img1 = cv2.imread(test_image1_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (256, 256))
    img1 = transforms.ToTensor()(img1)
    img1 = torch.unsqueeze(img1, dim=0)
    img2 = cv2.imread(test_image2_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, (256, 256))
    img2 = transforms.ToTensor()(img2)
    img2 = torch.unsqueeze(img2, dim=0)
    img = torch.cat((img1, img2), dim=0)
    label = torch.tensor([[0], [1]])

    # cutmix augmentation
    img, label_a, label_b, lam = cutmix(img, label, beta=1.0)
    img = img.permute((0, 2, 3, 1)).contiguous()
    img = img.numpy()
    print(label_a, label_b, lam)

    # figure plot
    f, axarr = plt.subplots(1, 2, figsize=(15,15))
    axarr[0].imshow(img[0])
    axarr[1].imshow(img[1])
    plt.show()
