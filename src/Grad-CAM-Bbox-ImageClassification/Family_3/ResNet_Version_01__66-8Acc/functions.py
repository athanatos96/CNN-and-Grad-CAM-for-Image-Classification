import cv2
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset



# Function to Import the data
def import_train_folder_dataset(root_path):
    class_names = os.listdir(root_path)
    
    img_data_files=[]
    label_data_files=[]
    for pos, img_class in enumerate(class_names):
        for img_name in os.listdir(os.path.join(root_path,img_class)):
            path_= os.path.join(root_path,img_class,img_name)
            
            img = cv2.imread(path_)
            img_data_files.append(img)
            label_data_files.append(pos)
    return( np.array(img_data_files),np.array(label_data_files) )

# Function to Import the data
def import_test_folder_dataset(root_path):
    #class_names = os.listdir(root_path)
    
    img_data_files=[]
    img_data_names=os.listdir(root_path)
    for img in img_data_names:
        path_= os.path.join(root_path,img)
            
        img = cv2.imread(path_)
        img_data_files.append(img)
    return( (np.array(img_data_files),np.array(img_data_names)) )


# Create the dataset object
class Data(Dataset):
    def __init__(self, X_train, y_train, transform):
        # convert to tensor
        self.X = X_train
        # Convert to tensor
        self.y = torch.from_numpy(y_train).type(torch.LongTensor)
        
        self.transform = transform
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        item = self.X[index]
        item = self.transform(item)
        return item, self.y[index]
    
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

    
    
    
    
    
def plot_loss_accuracy(train_loss, val_loss, train_accuracy, val_accuracy):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,6))

    fig.suptitle('Horizontally stacked subplots')
    ax1.plot(train_loss, label="Train_loss")
    ax1.plot(val_loss, label="Validation_loss")
    ax1.title.set_text("Loss")
    ax1.legend(loc="best")

    ax2.plot(train_accuracy, label="Train_Accuracy")
    ax2.plot(val_accuracy, label="Validation_Accuracy")
    ax2.title.set_text("Accuracy")
    ax2.legend(loc="best")

    plt.show()
    
    
    # create figure and axis objects with subplots()
    fig,ax = plt.subplots(figsize=(15,6))
    # make a plot
    ax.plot(train_loss, color="red", marker="o", label="Train_loss")
    ax.plot(val_loss, color="orange", marker="o", label="Validation_loss")
    # set x-axis label
    ax.set_xlabel("Epoch", fontsize = 14)
    # set y-axis label
    ax.set_ylabel("Loss Function",
                  color="red",
                  fontsize=14)


    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(train_accuracy,color="blue",marker="^", label="Train_Accuracy")
    ax2.plot(val_accuracy,color="green",marker="^", label="Validation_Accuracy")

    ax2.set_ylabel("Accuracy",color="blue",fontsize=14)

    ax.legend(loc="center left")
    ax2.legend(loc="center right")
    plt.show()   
    
# Save model Checkpoint    
def save_model(epochs, time, model, optimizer, criterion, path):
    """
    Function to save the trained model to disk.
    """
    # Remove the last model checkpoint if present.
    torch.save({
                'epoch': epochs+1,
                'time': time,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, path)
    
def save_metrics(train_loss, val_loss, train_accuracy, val_accuracy, path):
    # Method to save the results as a csv. Method by Alejandro C Parra Garcia
    dict = {'train_loss': train_loss, 'val_loss': val_loss, 'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy}  
    df = pd.DataFrame(dict)  
    df.to_csv(path)
    
    
# Save Predictions
def save_predictions_as_csv(names, predictions, name="placeholder.csv"):
    df = pd.DataFrame(list(zip(names, predictions)))#, columns =['Name', 'class']
    #save
    df.to_csv(name, sep=';', header=False,index=False )
