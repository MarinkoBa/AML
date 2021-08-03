import torch
import os
from utils.dataloader import ChestXRayDataset

if __name__ == "__main__":


    data_path = ...  # insert absolute path to the data directory
    train_test = 'train'  # train or test
    target_resolution = (128, 128)  # modify here if other resolution needed
    batch_size = 32

    dataset = ChestXRayDataset(os.path.join(data_path, train_test), target_resolution)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # label 0 -> normal | label 1 -> covid | label 2 -> pneumonia
    for idx, (data, label) in enumerate(dataloader):
        print(idx)
