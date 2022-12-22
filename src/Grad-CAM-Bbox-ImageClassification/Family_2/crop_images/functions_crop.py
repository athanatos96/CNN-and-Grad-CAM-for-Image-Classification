import cv2
import os
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset



# Function to Import the data
def import_train_folder_dataset(root_path):
    class_names = os.listdir(root_path)
    
    img_data_files=[]
    label_data_files=[]
    img_names_files = []
    for pos, img_class in enumerate(class_names):
        for img_name in os.listdir(os.path.join(root_path,img_class)):
            path_= os.path.join(root_path,img_class,img_name)
            
            img = cv2.imread(path_)
            #resized = cv2.resize(img, resized_side)

            #img_data_files.append(resized)
            img_data_files.append(img)
            label_data_files.append(pos)
            img_names_files.append(img_name)
    return( np.array(img_data_files),np.array(label_data_files),np.array(img_names_files) )

# Function to Import the data
def import_test_folder_dataset(root_path):
    #class_names = os.listdir(root_path)
    
    img_data_files=[]
    img_data_names=os.listdir(root_path)
    for img in img_data_names:
        path_= os.path.join(root_path,img)
            
        img = cv2.imread(path_)
        #resized = cv2.resize(img, resized_side)
        #img_data_files.append(resized)
        img_data_files.append(img)
    return( (np.array(img_data_files),np.array(img_data_names)) )


# Create the dataset object
class Data(Dataset):
    def __init__(self, X_train, y_train, names_train, transform):
        # convert to tensor
        self.X = X_train
        # Convert to tensor
        self.y = torch.from_numpy(y_train).type(torch.LongTensor)
        
        self.name = names_train
        
        self.transform = transform
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        item = self.X[index]
        item = self.transform(item)
        return item, self.y[index], self.name[index]
    
# Create the dataset object
class Data_test(Dataset):
    def __init__(self, X_train, y_train, transform):
        # convert to tensor
        self.X = X_train
        # Convert to tensor
        self.y = y_train
        
        self.transform = transform
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        item = self.X[index]
        item = self.transform(item)
        return item, self.y[index]

    
    
    
    
    
    

