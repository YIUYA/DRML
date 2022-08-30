import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import transforms
def pil_loader(path):

    with Image.open(path) as img:
        return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)



def make_dataset(image_list, label_list):
    len_ = len(image_list)
    images = [(image_list[i].strip(),  label_list[i, :]) for i in range(len_)]
    return images



class DISFA(Dataset):
    def __init__(self, root_path, train=True, crop_size = 170, loader=default_loader):

        self._root_path = root_path
        self._train = train
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = os.path.join(root_path,'img')
        self.transforms = transforms.ToTensor()
        if self._train:
            # img
            train_image_list_path = os.path.join(root_path, 'list', 'DISFA+_train_img_path.txt')
            train_image_list = np.loadtxt(train_image_list_path, dtype='str')
            # img labels
            train_label_list_path = os.path.join(root_path, 'list', 'DISFA+_train_label_12au_01.txt')
            train_label_list = np.loadtxt(train_label_list_path)

            self.data_list = make_dataset(train_image_list, train_label_list)

        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'DISFA+_test_img_path.txt')
            test_image_list = np.loadtxt(test_image_list_path, dtype='str')
            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'DISFA+_test_label_12au_01.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):

        img, label = self.data_list[index]
        img = self.loader(os.path.join(self.img_folder_path,img))

        img = img.resize((self.crop_size, self.crop_size), Image.ANTIALIAS)
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.data_list)
