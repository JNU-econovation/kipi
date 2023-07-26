import faiss
import json
import numpy as np
import torch

# step1. get user image 
# step2. find similar upper or onepiece in server closet (find cloth id)
# step3. using cloth id, find pants, outer(if exist)
# step4. find similar pants, outer in user closet
# step5. return cloth ids

def getRecommendCloth(user_img_path):
    # step1. find similar upper or onepiece in server closet (find cloth id)
    recommend_id = RecommendCloth('../data/prepro_data/upper', user_img_path)
    
    # step2. using cloth id, find pants, outer(if exist)
    with open('../data/kfashion_style/category_custom_final.json', 'r') as f:
        category_dict = json.load(f)
    
    category = category_dict[recommend_id]
    
    if category == 'upper':
        recommend_id = RecommendCloth('../data/prepro_data/pants', user_img_path)
    elif category == 'onepiece':
        recommend_id = RecommendCloth('../data/prepro_data/outer', user_img_path)
    
    # step3. find similar pants, outer in user closet
    recommend_id = RecommendCloth('../data/prepro_data/pants', user_img_path)
    
    # step4. return cloth ids
    return recommend_id

def RecommendCloth(base_closet_path, que_cloth_path):
    base_vec, base_id = getVecId(base_closet_path)
    que_vec, que_id = getVecId(que_cloth_path)
    
    base = faiss.IndexFlatL2(base_vec.shape[1])
    base = faiss.IndexIDMap2(base)
    
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, base)
        gpu_index_flat.add_with_ids(base_vec, np.arange(len(base_id)))
        
        D, I = gpu_index_flat.search(que_vec, 4)
    else:
        D, I = base.search(que_vec, 4)
    
    recommend_id = base_id[I[0][0][0]]
    
    return recommend_id

def getVecId(closet_path):
    pass