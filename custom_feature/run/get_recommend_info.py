import faiss
import json
import numpy as np
import torch

# step1. preprocessing user image data
# step2. find similar upper or onepiece in server closet (find cloth id)
# step3. using cloth id, find pants, outer(if exist)
# step4. find similar pants, outer in user closet
# step5. return cloth ids

def getRecommendCloth(user_closet, cur_temperture):
    user_vecs, user_ids, user_temperatures = preprocessing(user_closet)
    server_vecs, server_ids = getVecId('../data/server_data.json')
    # choose cloth that temperature is similar to current temperature and fac is upper or onepiece
    init_user_vecs = []
    init_user_ids = []
    init_user_facs = []
    
    for fac in ['upper', 'onepiece']:
        for i in range(len(user_vecs[fac])):
            if abs(user_temperatures[fac][i] - cur_temperture) < 2:
                    init_user_vecs.append(user_vecs[fac][i])
                    init_user_ids.append(user_ids[fac][i])
                    init_user_facs.append(fac)
    
    if len(init_user_vecs) == 0:
        print('error')
        return -1
    
    # choose random cloth in user_vecs
    rand_idx = np.random.randint(len(init_user_vecs))
    user_vec = init_user_vecs[rand_idx]
    user_id = init_user_ids[rand_idx]
    user_fac = init_user_facs[rand_idx]
    
    print(user_fac)
    
    ret_recommend_cloth = [{}]

    server_recommend_cloth_id = RecommendCloth((server_vecs[user_fac], server_ids[user_fac]), (user_vec, user_id))
    
    ret_recommend_cloth[0][user_fac] = user_id
    
    for fac in fac_list:
        if fac == user_fac:
            continue
        if server_recommend_cloth_id in server_ids[fac]:
            server_recommend_cloth_idx = server_ids[fac].index(server_recommend_cloth_id)
            ret_recommend_cloth[0][fac] = RecommendCloth((user_vecs[fac], user_ids[fac]), (server_vecs[fac][server_recommend_cloth_idx], None))
    
    print(ret_recommend_cloth)
    return ret_recommend_cloth
            
    

def RecommendCloth(base_closet, que_cloth):
    base_vec, base_id = np.array(base_closet[0]).astype('float32'), base_closet[1]
    que_vec, que_id = np.array(que_cloth[0]).astype('float32'), que_cloth[1]
    
    que_vec = que_vec.reshape(1, -1)
    
    print(base_vec.shape, que_vec.shape)
    
    base = faiss.IndexFlatL2(base_vec.shape[1])
    base = faiss.IndexIDMap2(base)
    
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, base)
        gpu_index_flat.add_with_ids(base_vec, np.arange(len(base_id)))
        
        D, I = gpu_index_flat.search(que_vec, 4)
    else:
        D, I = base.search(que_vec, 4)
    
    recommend_id = base_id[I[0][0]]
    
    return recommend_id

def getVecId(closet_path):
    ret_vecs = {}
    ret_ids = {}
    
    with open(closet_path, 'rb') as f:
        closet = json.load(f)
    
    fac_list = list(closet.keys())
    for fac in fac_list:
        ret_vecs[fac] = []
        ret_ids[fac] = []
    
    for fac in fac_list:
        cloth_id_list = list(closet[fac].keys())
        for cloth_id in cloth_id_list:
            ret_vecs[fac].append(closet[fac][cloth_id])
            ret_ids[fac].append(cloth_id)
            
    return ret_vecs, ret_ids
            
def preprocessing(user_closet):
    user_vecs_list = {}
    user_ids_list = {}
    user_temperture_list = {}
    
    fac_list = ['outer', 'upper', 'pants', 'onepiece']
    for fac in fac_list:
        user_vecs_list[fac] = []
        user_ids_list[fac] = []
        user_temperture_list[fac] = []
        
    for i in range(len(user_closet)):
        user_vecs_list[user_closet[i].fac].append(user_closet[i].feature)
        user_ids_list[user_closet[i].fac].append(user_closet[i].id)
        user_temperture_list[user_closet[i].fac].append(user_closet[i].temperture)
    
    return user_vecs_list, user_ids_list, user_temperture_list
    
    
class Cloth:
    def __init__(self, id, fac, feature, temperture):
        self.id = id
        self.feature = feature
        self.temperture = temperture
        self.fac = fac
    
    def __str__(self):
        return f'id: {self.id}, feature: {self.feature}, temperture: {self.temperture}'
    

if __name__ == '__main__':
    fac_list = ['outer', 'upper', 'pants', 'onepiece']
    trash_input = [Cloth(i, fac_list[np.random.randint(4)], [np.random.randint(10)/10. for _ in range(122)], 7) for i in range(30)]
    trash_temperture = 7
    
    getRecommendCloth(trash_input, trash_temperture)
    
    