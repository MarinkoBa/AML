import torch
from torchvision import transforms
from PIL import Image
import glob
import numpy as np
import utils.augmentator as augmentator


class ChestXRayDataset(torch.utils.data.Dataset):
    """Chest X Ray dataset."""

    def __init__(self, data_path, target_resolution, train_eval_test, balancing_mode):
        """
        Args:
            data_path (string): Path to dir with all the data.
            target_resolution (tuple): Tuple containing target resolution of the input images (HxW).
            train_eval_test (string, train or eval): if train, contains data used for the training, if eval, eval mode on, if test, test mode on.
            balancing_mode (string, no, balance_without_aug, balance_with_aug): if no, no balancing applied (all original data used)
                                                                                if balance_without_aug -> use undersampling
                                                                                if balance_with_aug -> use augmentation in order to increase dataset.
        """

        self.data_path = data_path
        all_images_file_path = glob.glob(self.data_path + "/*/*")
        self.transform = transforms.Compose([transforms.Resize(target_resolution), transforms.ToTensor()])

        self.all_images = []
        for i in all_images_file_path:
            img = Image.open(i)
            if img.mode != 'L' and img.mode != 'P' and  ").jpg" in i:  # there are L, RGB and RGBA modes in the data (we want RGB and RGBA)
                self.all_images.append(i)

        np.random.seed(0)  # use always same seed

        # get all RGB or RGBA image paths from the corresponding sub dataset
        self.normal_imgs = [norm_img for norm_img in self.all_images if "NORMAL" in norm_img]
        self.covid_imgs = [cov_img for cov_img in self.all_images if "COVID19" in cov_img]
        self.pneumonia_imgs = [pne_img for pne_img in self.all_images if "PNEUMONIA" in pne_img]

        if train_eval_test != 'test':
            # handle balancing
            if balancing_mode == 'balance_without_aug':
                self.all_images = augmentator.balance_dataset_undersampling(self.normal_imgs, self.covid_imgs, self.pneumonia_imgs)
                print("Balancing without augmentation activated...")
            elif balancing_mode == 'balance_with_aug':
                self.all_images = augmentator.augment_data(self.normal_imgs, self.covid_imgs, self.pneumonia_imgs, plot=False)
                print("Balancing with augmentation activated...")
            print("Total amount of data (sum train+eval): " + str(len(self.all_images)))

        np.random.shuffle(self.all_images)
        print("Shuffling dataset in progress...")


        if train_eval_test == 'train':
            self.all_images = self.all_images[:int(len(self.all_images)*0.7)]
            print("Total amount of training data (70% of all data): " + str(len(self.all_images)))

        elif train_eval_test == 'eval':
            self.all_images = self.all_images[int(len(self.all_images)*0.7):]
            print("Total amount of eval data (30% of all data): " + str(len(self.all_images)))
        else:
            print("Total amount of test data: " + str(len(self.all_images)))




    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):

        data_jpeg = Image.open(self.all_images[idx])
        data = self.transform(data_jpeg)
        if data_jpeg.mode == 'RGBA':
            data = data[:3, :, :]

        if "NORMAL" in self.all_images[idx]:
            label = 0
        elif "COVID19" in self.all_images[idx]:
            label = 1
        elif "PNEUMONIA" in self.all_images[idx]:
            label = 2

        return data, label
