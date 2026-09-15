import glob
import os
from PIL import Image
from torch.utils.data import Dataset

class Batset(Dataset):
    """
    Adaptation of PyTorch Dataset class for loading bat images from the file system.
    """

    def __init__(self, path, transform=None):
        """
        Loads the bat images from the specified path in the file system.

        :param path: Root folder of the dataset. The dataset is expected to contain separate class folders (e.g. different individuals) that contain images that belong together (e.g. all images from one approach of a bat).
        The names of the folders do not matter.
        :param transform: Optional transform you want to apply on the images when they are retrieved through __getitem__().
        """

        print(f"load dataset from {os.path.abspath(path)}")

        self.path = path
        self.transform = transform
        class_folders = sorted(glob.glob(self.path + "/*"), key=lambda name: int(os.path.basename(name)))
        self.data = []
        self.class_map = {}

        class_counter = 0

        for class_path in class_folders:
            individual_index = int(os.path.basename(class_path))

            self.class_map[individual_index] = class_counter

            
            for img_path in glob.glob(os.path.join(class_path, '**', '*.png'), recursive=True):
                self.data.append((img_path, individual_index))

            class_counter += 1

        print(f"loaded {len(self.data)} images within {class_counter} classes")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, individual_index = self.data[idx]
        class_id = self.class_map[individual_index]
        
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, class_id, img_path