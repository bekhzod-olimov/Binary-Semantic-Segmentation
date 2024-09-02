# Import libraries
import torch, sys, torchvision, os, cv2, albumentations as A
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as tfs
from src.transformations import get_transformations
from src.create_data import create_data
from glob import glob
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(2023)
sys.path.append("./")

class CustomSegmentationDataset(Dataset):
        
    # Initialization
    def __init__(self, ds_name, transformations = None, im_files = [".jpg", ".png", ".jpeg"]):

        self.transformations = transformations
        self.tensorize = tfs.Compose([tfs.ToTensor()]) 
        root = "datasets/cells" if ds_name == "cells" else ("datasets/flood" if ds_name == "flood" else "datasets/drone")
        if not os.path.isdir(root): create_data(save_dir = "datasets", data_name = ds_name)
        self.threshold = 11 if ds_name == "drone" else 128
        self.im_paths = sorted(glob(f"{root}/images/*[{im_file for im_file in im_files}]"))
        self.gt_paths = sorted(glob(f"{root}/masks/*[{im_file for im_file in im_files}]"))
    
    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):

        im, gt = cv2.cvtColor(cv2.imread(self.im_paths[idx]), cv2.COLOR_BGR2RGB), cv2.cvtColor(cv2.imread(self.gt_paths[idx]), cv2.COLOR_BGR2GRAY)
        if self.transformations is not None: 
            transformed = self.transformations(image = im, mask = gt)
            im, gt = transformed['image'], transformed['mask']
            
        return self.tensorize(im), torch.tensor(gt > self.threshold).long()
    
    
def get_dl(ds_name, transformations, bs, split = [0.7, 0.15, 0.15]):
        
    assert sum(split) == 1., "Sum of data split must be equal to 1"
    
    ds = CustomSegmentationDataset(ds_name = ds_name, transformations = transformations)
    
    tr_len = int(len(ds) * split[0])
    val_len = int(len(ds) * split[1])
    test_len = len(ds) - (tr_len + val_len)
    
    # Data split
    tr_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [tr_len, val_len, test_len])
        
    print(f"\nThere are {len(tr_ds)} number of images in the train set")
    print(f"There are {len(val_ds)} number of images in the validation set")
    print(f"There are {len(test_ds)} number of images in the test set\n")
    
    # Get dataloaders
    tr_dl  = DataLoader(dataset = tr_ds, batch_size = bs, shuffle = True, num_workers = 8)
    val_dl = DataLoader(dataset = val_ds, batch_size = bs, shuffle = False, num_workers = 8)
    test_dl = DataLoader(dataset = test_ds, batch_size = 1, shuffle = False, num_workers = 8)
    
    return tr_dl, val_dl, test_dl

# ts = get_transformations(224)[1]
# tr_dl, val_dl, test_dl = get_dl(ds_name = "flood", transformations = ts, bs = 2)