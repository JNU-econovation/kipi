import argparse
import torch

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.fashion_mnist import FashionMNIST
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

from utility.sam import SAM

import os

from PIL import Image
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from run.custom_feature_ext import custom_feature

import extcolors

import numpy as np

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

class2fac = {"T-shirt/top":"upper",
             "Trouser":"pants",
             "Pullover":"upper",
             "Dress":"onepiece",
             "Coat":"outer",
             "Sandal":"Undefined",
             "Shirt":"upper",
             "Sneaker":"Undefined",
             "Bag":"Undefined",
             "Ankle boot":"Undefined"
}

args = None

def get_args():
    global args
    parser = argparse.ArgumentParser(description='WILDCAT Training')
    parser.add_argument('--data', default='../data/')
    parser.add_argument('--wordvec', default='../data/kfashion_style/custom_glove_word2vec_final.pkl')
    parser.add_argument('--adj', default='../data/kfashion_style/custom_adj_final.pkl')
    parser.add_argument('--image-size', default=224, type=int)
    parser.add_argument('-j', '--workers', default=12, type=int)
    parser.add_argument('--device_ids', default=[0,1,2,3], type=int, nargs='+')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', default=True, action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.parse_args()

get_args()
def getImgFac(img_path):
    global args
    
    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=1, labels=10).to(device)
    model.load_state_dict(torch.load('../checkpoint/sam_60.pth'))
    model.eval()
    
    ret = str()
    
    convert_tensor = transforms.ToTensor()

    plt_img = Image.open(img_path)

    img_tensor = convert_tensor(plt_img)

    image_trans = transforms.Compose([
                                        transforms.Grayscale(),
                                        transforms.Resize((28,28)),
                                        transforms.Normalize(torch.mean(img_tensor), torch.std(img_tensor),inplace=True)
                                    ])

    img_tensor_transformed = image_trans(img_tensor).unsqueeze(0).to(device)

    ret = classes[int(torch.argmax(model(img_tensor_transformed).data,1))]
    ret = class2fac[ret]

    return ret


def getImgFeature(img_path):
    global args

    feature_list = ['style', 'category', 'texture','detail', 'print']
    feature_weight = {'color':7,'style':5, 'category':1, 'texture':1,'detail':4, 'print':2}
    test_vec_list = []

    img = Image.open(img_path)
    width, height = img.size
    img = img.crop((width/8, height/8, width*7/8, height*7/8))

    color_que, _ = extcolors.extract_from_image(img)
    color_que = list(color_que[0][0])
    color_que = torch.tensor(color_que)/255.0/3.0*feature_weight['color']
    test_vec_list.append(color_que)

    for feature in feature_list:
        que, _ = custom_feature(feature, args, phase='test', img_path=img_path)
        print(que[1])
        que = torch.tensor(que[0]).squeeze()
        # print(que.shape)
        que = que/torch.max(que)
        test_vec_list.append(que/float(que.shape[0])*feature_weight[feature])

    test_vec = np.reshape(torch.cat(test_vec_list,dim=0).numpy(),(1,-1))
    test_vec[test_vec<0]=0
    test_vec = test_vec.tolist()

    return test_vec[0]


def getCrossInfo(img_path):
    return (getImgFac(img_path), getImgFeature(img_path))


if __name__ == '__main__':
    img_path = '../img/test/test_shirts2.jpg'
    print(getCrossInfo(img_path))