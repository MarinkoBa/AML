import torch
from torchvision import transforms
from PIL import Image
import glob


class ChestXRayDataset(torch.utils.data.Dataset):
    """Chest X Ray dataset."""

    def __init__(self, data_path, target_resolution):
        """
        Args:
            data_path (string): Path to dir with all the data.
        """

        self.data_path = data_path
        all_images_file_path = glob.glob(self.data_path + "/*/*")
        self.transform = transforms.Compose([transforms.Resize(target_resolution), transforms.ToTensor()])

        self.all_images = []
        for i in all_images_file_path:
            img = Image.open(i)
            if img.mode == 'RGB':
                self.all_images.append(i)


    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):

        data = Image.open(self.all_images[idx])
        data = self.transform(data)
        if "NORMAL" in self.all_images[idx]:
            label = 0
        elif "COVID19" in self.all_images[idx]:
            label = 1
        elif "PNEUMONIA" in self.all_images[idx]:
            label = 2

        return data, label
