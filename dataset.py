from torch.utils.data import Dataset
import os
import glob
from PIL import Image


class TripletDataSet(Dataset):
    _CLASSES_MAP = {'nomal':1, 'arms': 0}  # follow YOLO cls
    def __init__(self, root_path, transform=None):
        self.database = []
        for cls in os.listdir(root_path):
            # file xxxx.bmp
            for im_file in glob.glob(os.path.join(root_path, cls) + '/*.bmp'):
                self.database.append((im_file, TripletDataSet._CLASSES_MAP[cls]))
        self.transform = transform

    def __getitem__(self, idx):
        im_file, lb = self.database[idx]
        im = Image.open(im_file)
        im = im.convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, lb
        
    def __len__(self):
        return len(self.database)
    
    def get_labels(self):
        return [i[1] for i in self.database]