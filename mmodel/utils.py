import pickle
import os.path as osp 
import torch
import os 
import pandas as pd 
import random 
import numpy as np 


def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_esm(lookup):
    esm = torch.load(f"../data/esmdata/{lookup}.pt")
    return esm.unsqueeze(0)  # (1, embeddingDim)
def load_embedding(label_name_dict, device, dtype, path="../data/esmdata/"):
    """
    load_embedding 从本地读取 embedding

    Args:
        label_name_dict (_type_): _description_
        device (_type_): _description_
        dtype (_type_): _description_

    Returns:
        _type_: _description_
    """
    name_embedding_dict = {}

    flatten_list = [load_esm(name) for label, name_list in label_name_dict.items() for name in name_list]  # label1: [name1, name2, name3], label2: [name1, name2, name3] => [name1, name2, name3, name1, name2, name3]
        
    return torch.cat(flatten_list, dim=0).to(device=device, dtype=dtype)
            

def get_label_name_dict(csv_path, label="label", name="name"):
    csv_file = pd.read_csv(csv_path)

    label_name_dict = {}
    for idx, df in csv_file.iterrows():
        label_current = df[label]
        name_current = df[name]
        if label_current not in label_name_dict.keys():
            label_name_dict[label_current]=[name_current]
        else:
            label_name_dict[label_current].append(name_current)

    return label_name_dict

def get_name_label_dict(label_name_dict):
    name_label_dict ={}
    for label_current, name_current_list in label_name_dict.items():
        for name_current in name_current_list:
            if name_current not in name_label_dict.keys():
                name_label_dict[name_current]=[label_current]
            else:
                name_label_dict[name_current].append(label_current)

    return name_label_dict


