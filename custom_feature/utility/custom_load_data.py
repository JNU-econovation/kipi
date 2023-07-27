import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch.utils.data as data
import json
import os
import pickle
from utility.util import *
import numpy as np

class load_data(data.Dataset):
    def __init__(self, root, phase='custom', inp_name=None, img_path=None):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.img_path = img_path
        self.get_anno()

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)

        self.inp_name = inp_name

    def get_anno(self):
        if self.phase == 'test':
            self.img_list = [{'file_name':self.img_path}]
        else:
            list_path = os.path.join(self.root,'prepro_data/', '{}_img.json'.format(self.phase))
            self.img_list = json.load(open(list_path, 'r'))
            self.length = len(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']

        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        return (img, filename, self.inp)