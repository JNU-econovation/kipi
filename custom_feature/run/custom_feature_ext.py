import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pickle

from utility.custom_load_data import *
from utility.ml_gcn import *
from utility.custom_engine import *

from utility.resnest import *



def custom_feature(model_name, args, phase='custom', img_path=None):
    
    dataset = load_data(root = args.data, phase=phase, inp_name=args.wordvec, img_path=img_path)
    state = {'batch_size': args.batch_size, 'image_size': args.image_size}

    if model_name == 'style':
        state['num_classes'] = 10
        state['resume'] = '../checkpoint/kfashion_style/model_best.pth.tar'
        state['isstyle'] = True
        model = gcn_resnet101(num_classes=state['num_classes'], t=0.03, adj_file=args.adj)
    else:
        if model_name == 'category':
            state['num_classes'] = 21
            state['resume'] = '../checkpoint/kfashion_category/model_category_best.pth.tar'
        elif model_name == 'detail':
            state['num_classes'] = 40
            state['resume'] = '../checkpoint/kfashion_detail/model_detail_best.pth.tar'
        elif model_name == 'texture':
            state['num_classes'] = 27
            state['resume'] = '../checkpoint/kfashion_texture/model_texture_best.pth.tar'
        elif model_name == 'print':
            state['num_classes'] = 21
            state['resume'] = '../checkpoint/kfashion_print/model_print_best.pth.tar'
        state['isstyle'] = False
        model = resnest50d(pretrained=False, nc=state['num_classes'])

    
    state['workers'] = args.workers

    
    if args.evaluate:
        state['evaluate'] = True
    engine = Engine(state)
    
    
    return engine.learning(model, dataset), time.time()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='WILDCAT Training')
    parser.add_argument('--data', default='../data/')
    parser.add_argument('--wordvec', default='../data/kfashion_style/custom_glove_word2vec_final.pkl')
    parser.add_argument('--adj', default='../data/kfashion_style/custom_adj_final.pkl')
    parser.add_argument('--image-size', default=224, type=int)
    parser.add_argument('-j', '--workers', default=12, type=int)
    parser.add_argument('--device_ids', default=[0,1,2,3], type=int, nargs='+')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', default=True, action='store_true',
                        help='evaluate model on validation set')
    args = parser.parse_args()

    stt = time.time()
    feature_list = ['style', 'category', 'detail', 'texture', 'print']
    fac_list = ['outer', 'pants', 'onepiece', 'upper']

    for feature in feature_list:
        feature_ret = []
        feature_name = []
        for fac in fac_list:
            f_st = time.time()
            (a,b), t = custom_feature(feature, args, phase=fac)
            feature_ret.extend(a)
            feature_name.extend(b)
            
            print(f'{feature} feature spent time : {t-f_st}')
        with open(f'../data/prepro_data/{feature}_feature.pkl','wb') as f:
            pickle.dump(feature_ret,f)
        with open(f'../data/prepro_data/{feature}_feature_name.pkl','wb') as f:
            pickle.dump(feature_name,f)

        print(f'total spent time : {time.time()-stt}')
