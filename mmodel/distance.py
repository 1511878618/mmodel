import pickle
from tqdm import tqdm 
import torch
from .utils import get_name_label_dict, get_label_name_dict, ensure_dirs, load_embedding

def get_cluster_center(model_emb, label_name_dict):
    cluster_center_model = {}
    slice_counter = 0
    with torch.no_grad():
        for label in tqdm(list(label_name_dict.keys())):
            names_for_query = list(label_name_dict[label])
            slice_counter_prime = slice_counter + len(names_for_query)
            emb_cluster = model_emb[slice_counter: slice_counter_prime]
            cluster_center = emb_cluster.mean(dim=0)
            cluster_center_model[label] = cluster_center.detach().cpu()
            slice_counter = slice_counter_prime
    return cluster_center_model


def dist_func(q,k):
    """
    q: query, (1, embeddingDim)
    k: key, (N, embeddingDim)
    return:(N,) 
    """
    return torch.norm(q-k, dim=1, p=2)


def get_dist_map(label_name_dict, emb_tensor, device, dtype, dist_func=dist_func,model=None,):
    """
    label中的name展开后叠在一起的顺序即是返回的dist_map中的顺序
    get the distance map for training, all names vs names distance map
    {name1:{name1:dist1, name2:dist2, ...}, name2:{name1:dist1, name2:dist2, ...}, ...}
    """
    if model is not None:
        emb_tensor = model(emb_tensor.to(device=device, dtype=dtype))
    else:
        emb_tensor = emb_tensor
    name_label_dict = get_name_label_dict(label_name_dict)

    dist_dict = {}
    for i, name in enumerate(name_label_dict.keys()):
        current = emb_tensor[i].unsqueeze(0)
        dist = dist_func(current, emb_tensor).detach().cpu().numpy()
        dist_dict[name] = {name:dist[i] for i, name in enumerate(name_label_dict.keys())}
    return dist_dict 

def compute_emb_distance(dataset, issave=True):
    ensure_dirs('../data/distance_map/')
    name_label_dict = get_label_name_dict(f"../data/{dataset}.csv")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    emb_tensor = load_embedding(name_label_dict, device, dtype)
    emb_dist = get_dist_map(name_label_dict, emb_tensor, device, dtype)
    if issave:
        pickle.dump(emb_dist, open(
            '../data/distance_map/' + dataset + '.pkl', 'wb'))
        pickle.dump(emb_tensor, open('../data/distance_map/' +
                    dataset + '_emb.pkl', 'wb'))
    else:
        return emb_dist, emb_tensor
