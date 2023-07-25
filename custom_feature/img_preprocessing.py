import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_dir = 'img/clothes/'
json_dir = './img/labels/'
img_dir_dst = 'img/prepro_clothes/'

kor2eng = {'아우터' : 'outer',
           '하의' : 'pants',
           '원피스' : 'onepiece',
           '상의' : 'upper'}

files = os.listdir(img_dir)
files = list(map(lambda x : x.replace('.jpg', ''), files))

for i, file in enumerate(files):
    with open(json_dir+file+'.json', 'r') as f:
        json_data = json.load(f)
    for k, v in json_data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표'].items():
        rect = json_data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']
        poly = json_data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']
        if len(v[0]) == 0:
            continue
        img = cv2.imread(img_dir+file+'.jpg')
        rect_dict = {}
        rect_dict['x1'] = int(rect[k][0]['X좌표']) if int(rect[k][0]['X좌표']) > 0 else 0
        rect_dict['x2'] = int(rect[k][0]['X좌표'] + rect[k][0]['가로']) if int(rect[k][0]['X좌표'] + rect[k][0]['가로']) < img.shape[1] else img.shape[1]
        rect_dict['y1'] = int(rect[k][0]['Y좌표']) if int(rect[k][0]['Y좌표']) > 0 else 0
        rect_dict['y2'] = int(rect[k][0]['Y좌표'] + rect[k][0]['세로']) if int(rect[k][0]['Y좌표'] + rect[k][0]['세로']) < img.shape[0] else img.shape[0]

        # polyX_dict = {}
        # polyY_dict = {}

        # poly_arr = []
        # for a, b in poly[k][0].items():
        #     if 'X좌표' in a:
        #         polyX_dict[int(a.replace('X좌표', ''))] = int(b)
        #     else:
        #         polyY_dict[int(a.replace('Y좌표', ''))] = int(b)
        # temp_keys = list(polyX_dict.keys())
        # temp_keys.sort()
        # for seq in temp_keys:
        #     poly_arr.append([polyX_dict[seq], polyY_dict[seq]])
        # np_poly = np.array(poly_arr)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # masking image
#         mask = np.zeros_like(img)
#         mask = cv2.fillPoly(mask, [np_poly], (1,1,1))
#         img = img*mask
        img = img[rect_dict['y1']:rect_dict['y2'],rect_dict['x1']:rect_dict['x2']]
        try:  
            cv2.imwrite(f'{img_dir_dst}{kor2eng[k]}/{file}_{kor2eng[k]}.jpg', img)
        except:
            print(f'{file}.jpg has problem')
            print(rect)

    if i%100 == 0 :
        print(f'{i}th/{len(files)}th processing...')