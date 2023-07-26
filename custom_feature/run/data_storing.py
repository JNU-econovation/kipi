main_dict = dict()

fac_list = ['upper', 'pants', 'onepiece', 'outer']

import pickle
import torch
import json

for fac in fac_list:
    feature_list = ['color', 'style', 'category', 'texture','detail', 'print']
    feature_weight = {'color':7,'style':5, 'category':1, 'texture':1,'detail':4, 'print':2}

    db_list = []

    for feature in feature_list:
        if feature == 'color':
            with open(f'../data/prepro_data/{fac}/{feature}_feature.pkl', 'rb') as f:
                colors_list = pickle.load(f)
            colors_tensor = torch.stack((list(map(torch.tensor,colors_list))))/255.0/3.0*feature_weight['color']
            db_list.append(colors_tensor)
        else:
            with open(f'../data/prepro_data/{fac}/{feature}_feature.pkl', 'rb') as f:
                a = pickle.load(f)
            b_1 = torch.cat(list(map(torch.tensor,a)), dim = 0)
            b_1 = b_1/torch.max(b_1)/float(b_1.shape[1])*feature_weight[feature]
            db_list.append(b_1)
    
    db_vec = torch.cat(db_list, dim = 1).numpy()
    db_vec[db_vec<0]=0

    print(db_vec.shape)

    with open(f'../data/prepro_data/{fac}/category_feature_name.pkl', 'rb') as f:
        c = pickle.load(f)
    file_names = []
    for i in range(len(c)):
        file_names.extend(c[i])
    

    main_dict[fac] = {int(file_names[i].split('/')[4].split('_')[0]):db_vec[i].tolist() for i in range(len(file_names))}

print(main_dict.keys())

with open('../server_data.json', 'w') as f:
    json.dump(main_dict, f)

