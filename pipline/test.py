import sys
import os

sys.path.append("/p300s/wangmx_group/xutingfeng/project_development/mmodel")

from mmodel.dataloader import *
import torch 
def get_dataloader(dist_map, label_name_dict, name_label_dict,knn=3, params=None):
    if params is None:
        params = {'batch_size': 32, "shuffle": True, "num_workers": 8}
    else:
        params = {'batch_size':32, "shuffle": True, "num_workers": 8, **params}
    print(params)
    pos = mine_hard_positive(dist_map=dist_map, name_label_dict=name_label_dict, label_name_dict=label_name_dict, knn=3)
    neg = mine_hard_negative(dist_map=dist_map, name_label_dict=name_label_dict, label_name_dict=label_name_dict, knn=3)
    Triplets = Triplet_dataset(label_name_dict, name_label_dict, neg, pos)
    train_loader = torch.utils.data.DataLoader(Triplets, **params)
    return train_loader


dataname="dataset"  # 数据集需要保存在../data/dataname.csv

#读取标签数据
import pickle
from mmodel.utils import get_label_name_dict
label_name_dict = get_label_name_dict(f"../data/{dataname}.csv")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
dtype = torch.float32

print('device used:', device, '| dtype used: ',
          dtype,)
from mmodel.utils import get_name_label_dict
name_label_dict = get_name_label_dict(label_name_dict)

esm_emb = pickle.load(
    open('../data/distance_map/' +  dataname+ '_emb.pkl',
            'rb')).to(device=device, dtype=dtype)
dist_map = pickle.load(open('../data/distance_map/' + \
    dataname + '.pkl', 'rb')) 

dataloader = get_dataloader(dist_map, label_name_dict, name_label_dict, knn=3, params={"batch_size": 4, "shuffle": True, "num_workers": 4})

for data in dataloader:
    print(data)
    break 
