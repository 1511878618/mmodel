import pickle
import os.path as osp 
import torch
import os 
import pandas as pd 
import random 
import numpy as np 

"""
raw data: name, label, seq1, seq2... 列需要
"""

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
    esm = torch.load(f"../data/esm_data/{lookup}.pt")
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

def fusion(*vec, methods="mean"):

    if methods == "mean":
        return np.mean(vec, axis=0)
    elif methods == "sum":
        return np.sum(vec, axis=0)
    elif methods == "concat":
        return np.concatenate(vec, axis=0)


def embed_this(filepath, embedder, seq_col=["SLF_seq", "RNase_seq"]):
    """
    raw data: name, label, seq1, seq2... 列需要
    对CSV进行embedding， col，指定序列列名，name_col指定name列名, label_col指定label列
    返回一个dataframe，包含原始数据，以及embedding
    """
    total = pd.read_csv(filepath)
    for i in seq_col:
        embeddings = [embedder.reduce_per_protein(embedding) for embedding in embedder.embed_many(total[i])]
        total[f"{i}_embedding"] = pd.Series(embeddings)

    return total 

def csv_to_fusion_embedding(csvpath, embedder, seq_col=None,name_col=None, methods="mean", savedir=None):
    """
    读取csv文件，进行embedding，返回一个dataframe
    embedder：embedder from BioEmbedding
    seq_col: list of seq column name
    name_col: name column name
    methods: fusion methods, mean, sum, concat,minus
    savedir: if not None, save the embedding to savedir，推荐data/esmdata 供后续使用，这个是后续训练模型所必须的！，如果仅仅embedding则不用
    """
    embed_csv = embed_this(csvpath, embedder, seq_col=seq_col)
    if savedir:
        ensure_dirs(savedir)
    embedding = []
    for idx, df in embed_csv.iterrows():
        fusion_tensor = torch.tensor(fusion(*[df[f"{i}_embedding"] for i in seq_col], methods=methods))
        print(embedding)
        if savedir:
            torch.save(fusion_tensor, osp.join(savedir,f"{df[name_col]}.pt"))
        embedding.append(fusion_tensor.unsqueeze(0))
    return torch.cat(embedding, dim=0)




