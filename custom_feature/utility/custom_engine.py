import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from utility.util import *
import json
import pandas as pd

tqdm.monitor_interval = 0
class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 25

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        if self._state('use_pb') is None:
            self.state['use_pb'] = True

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_forward(self, model):
        feature_var = torch.autograd.Variable(self.state['feature']).float()
        inp_var = torch.autograd.Variable(self.state['input']).float().detach()  # one hot

        with torch.no_grad():
            # compute output
            if self.state['isstyle'] : 
                self.state['output'] = model(feature_var, inp_var)
            else :
                self.state['output'] = model(feature_var)

            
    def on_start_batch(self):

        input = self.state['input']
        self.state['feature'] = input[0]
        self.state['out'] = input[1]
        self.state['input'] = input[2]
        

    def init_learning(self, model):

        if self._state('val_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])

    def learning(self, model, dataset):

        self.init_learning(model)

        # define train and val transform
        dataset.transform = self.state['val_transform']
        dataset.target_transform = self._state('val_target_transform')

        # data loading code
        data_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(self.state['resume'], map_location=device)
                model.load_state_dict(checkpoint['state_dict'])
                
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))


        if self.state['use_gpu']:
            data_loader.pin_memory = True
            cudnn.benchmark = True

            model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()

        if self.state['evaluate']:
            a, b= self.predict(data_loader, model)
            # return shape = -1, batch, classes
            return a, b


    def predict(self, data_loader, model):
            # switch to evaluate mode
            model.eval()
            a = []
            b = []
            data_len = len(data_loader)

            for i, (input) in enumerate(data_loader):
                # measure data loading time
                self.state['input'] = input
                self.on_start_batch()
                self.on_forward(model)
                a.append(self.state['output'].tolist())
                b.append(self.state['out'])
                if i%10==0:
                    print(f'{i}th/{data_len}th progress...')
            return a, b

