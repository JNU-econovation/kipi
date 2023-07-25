import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse

import faiss
import pickle
import torch
import json

import torch

from PIL import Image

import extcolors

import numpy as np
from custom_feature_ext import custom_feature


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
args = parser.parse_args()

def get_img_feature(img_path) -> np.array():
    test_vec_list = []

    img = Image.open(img_path)
    width, height = img.size
    img = img.crop((width/8, height/8, width*7/8, height*7/8))

    color_que, _ = extcolors.extract_from_image(img)
    color_que = list(color_que[0][0])
    color_que = torch.tensor(color_que)/255.0/3.0*feature_weight['color']
    test_vec_list.append(color_que)

    for feature in feature_list:
        que, _ = custom_feature(feature, args, phase='test')
        print(que[1])
        que = torch.tensor(que[0]).squeeze()
        # print(que.shape)
        que = que/torch.max(que)
        test_vec_list.append(que/float(que.shape[0])*feature_weight[feature])

    test_vec = np.reshape(torch.cat(test_vec_list,dim=0).numpy(),(1,-1))
    test_vec[test_vec<0]=0

    return test_vec

def faiss_feature(img_dir, fac, isfirst):
    
    feature_list = ['style', 'category', 'texture','detail', 'print']
    feature_weight = {'color':7,'style':5, 'category':1, 'texture':1,'detail':4, 'print':2}
    feature_list_doesn_use = []

    feature_dict = {}

    for feature in feature_list:
        if feature == 'style':
            with open('../data/kfashion_style/category_custom_final.json', 'r') as f:
                feature_dict[feature] = {v:k for k,v in json.load(f).items()}
        else:
            with open(f'../data/kfashion_{feature}/category_{feature}_final2.json', 'r') as f:
                feature_dict[feature] = {v:k for k,v in json.load(f).items()}
        
    with open(f'../data/prepro_data/{fac}/category_feature_name.pkl', 'rb') as f:
        c = pickle.load(f)
    file_names = []
    for i in range(len(c)):
        file_names.extend(c[i])

    db_list = []

    with open(f'../data/prepro_data/{fac}/color_feature.pkl', 'rb') as f:
        colors_list = pickle.load(f)

    colors_tensor = torch.stack((list(map(torch.tensor,colors_list))))/255.0/3.0*feature_weight['color']

    db_list.append(colors_tensor)

    for feature in feature_list:
        with open(f'../data/prepro_data/{fac}/{feature}_feature.pkl', 'rb') as f:
            a = pickle.load(f)
        b_1 = torch.cat(list(map(torch.tensor,a)), dim = 0)
        b_1 = b_1/torch.max(b_1)
        db_list.append(b_1/float(b_1.shape[1])*feature_weight[feature])
    
    db_vec = torch.cat(db_list, dim = 1).numpy()
    db_vec[db_vec<0]=0

    if isfirst:
        test_vec = test_vec_gen()

    
    Index = faiss.IndexFlatL2(db_vec.shape[1])
    Index = faiss.IndexIDMap2(Index)
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, Index)
        gpu_index_flat.add_with_ids(db_vec, np.arange(len(file_names)))
        
        D, I = gpu_index_flat.search(test_vec, 4)
    else:
        D, I = Index.search(test_vec, 4)

    a = D[0][:5].tolist()
    b = [file_names[g] for g in I[0][:5]]

    print(a)
    print(b)

    

if __name__ == '__main__':
    faiss_feature()