import argparse
import torch

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.fashion_mnist import FashionMNIST
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

import sys; sys.path.append("..")
from sam import SAM

import os

from PIL import Image
from torchvision import transforms

classes = ( "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot")



def classClassifier(plt_img):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=70, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--save_freq", default=10, type=int, help="model save frequency.")
    args,_ = parser.parse_known_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=1, labels=10).to(device)
    model.load_state_dict(torch.load('./model_save/sam_60.pth'))
    model.eval()
    
    ret = str()
    

    convert_tensor = transforms.ToTensor()
    img_tensor = convert_tensor(plt_img)
    img_tensor = torch.max(img_tensor)-img_tensor

    image_trans = transforms.Compose([
                                        transforms.Grayscale(),
                                        transforms.Resize((28,28)),
                                        transforms.Normalize(torch.mean(img_tensor), torch.std(img_tensor),inplace=True)
                                    ])

    img_tensor_transformed = image_trans(img_tensor).unsqueeze(0).to(device)
    
#     return img_tensor_transformed
    
#     print(model(img_tensor_transformed))

    ret = classes[int(torch.argmax(model(img_tensor_transformed).data,1))]


    return ret
