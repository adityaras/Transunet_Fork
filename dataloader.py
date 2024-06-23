import os
import numpy as np
import pandas as pd
from torchvision import transforms
import skimage.io as io
import skimage
from skimage.morphology import skeletonize, dilation, diamond
from torch.utils.data import Dataset, DataLoader
import torch
import cv2


class LoadData(Dataset):
    def __init__(self, fileNames, rootDir, dilate_skel=False, double_channel = False,transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.dc = double_channel
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=',', header=None)
        self.dilate_skel = dilate_skel
    
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):

        inputName = os.path.join(self.rootDir, self.frame.iloc[idx, 0][1:])
        targetName = os.path.join(self.rootDir, self.frame.iloc[idx, 1][1:])
        inputImage = cv2.imread(inputName)
        targetImage = cv2.imread(targetName, cv2.IMREAD_GRAYSCALE)
        
        targetImage = targetImage > 0.0
        print("Dilate skel: "+self.dilate_skel)
        if self.dilate_skel:
            dilated_target = skeletonize(targetImage)
            dilated_target = dilation(targetImage, diamond(2))
            dilated_target = dilated_target.astype(np.float32)
            dilated_target = np.expand_dims(dilated_target,axis=0)
                
        counts = np.unique(targetImage,return_counts=True)[1]
        weights = np.array([ counts[0]/(counts[0]+counts[1]) , counts[1]/(counts[0]+counts[1]) ])
        inputImage = inputImage.astype(np.float32)

        targetImage = targetImage.astype(np.float32)
        inputImage = inputImage.transpose((2, 0, 1))

        targetImage = np.expand_dims(targetImage,axis=0)
        
        if self.dilate_skel:
            return inputImage, targetImage,weights, dilated_target, self.frame.iloc[idx, 0]
        else:
            return inputImage, targetImage,weights, self.frame.iloc[idx, 0]
