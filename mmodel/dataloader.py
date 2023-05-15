import random 
import torch 

def mine_hard_negative(dist_map,name_label_dict, label_name_dict, knn = 3):
    """
    获取hard negative
    从dist_map中获取knn个hard negative
    hard negative: 与query不同的label的样本，且距离最近的knn个样本，这些样本容易混淆，因此作为hard negative
    """
    name_list = list(dist_map.keys())
    negative = {} 

    for i, query in enumerate(name_list):
        query_label = name_label_dict[query]
        query_dist = dist_map[query].copy()  # 复制字典，避免后续修改原始dist_map
        # filter same label
        for k in list(query_dist.keys()):
            if name_label_dict[k] == query_label:  
                query_dist.pop(k)  # 剔除同label的样本
        sort_dist_map = sorted(query_dist.items(), key=lambda x: x[1])  # 对距离排序

        freq = [1/i[1] for i in sort_dist_map[0:knn] ]  # 最近的样本更大概率被选中
        normalized_freq = [i/sum(freq) for i in freq]

        neg_names = [i[0] for i in sort_dist_map[0:knn]]
        negative[query] = {
            'weights': normalized_freq,
            'negative': neg_names
        }
    return negative
def mine_hard_positive(dist_map,name_label_dict, label_name_dict, knn = 3):
    """
    获取hard positive
    从dist_map中获取knn个hard positive
    hard positive: 与query相同的label的样本，且距离最近的knn个样本，这些样本容易混淆，因此作为hard positive
    """

    name_list = list(dist_map.keys())
    positive = {} 

    for i, query in enumerate(name_list):
        query_label = name_label_dict[query]
        query_dist = dist_map[query].copy()  # 复制字典，避免后续修改原始dist_map
        # filter same label
        for k in list(query_dist.keys()):
            if name_label_dict[k] != query_label:  # 筛选同label的pos样本
                query_dist.pop(k)
        sort_dist_map = sorted(query_dist.items(), key=lambda x: x[1],reverse=True)  # 寻找距离最远的同类

        freq = [i[1] for i in sort_dist_map[0:knn] ]  # 不取倒数了，是因为希望距离大的点后续被sample的概率更大
        normalized_freq = [i/sum(freq) for i in freq]

        pos_names = [i[0] for i in sort_dist_map[0:knn]]
        positive[query] = {
            'weights': normalized_freq,
            'positive': pos_names
        }
    return positive



def mine_negative(anchor, mine_neg):

    neg_names_list = mine_neg[anchor]['negative']
    weights = mine_neg[anchor]['weights']

    negative_select = random.choices(neg_names_list, weights=weights, k=1)[0]
    return negative_select


def mine_positive(anchor, mine_pos):
    pos_names_list = mine_pos[anchor]['positive']
    weights = mine_pos[anchor]['weights']
    positive_select = random.choices(pos_names_list, weights=weights, k=1)[0]
    return positive_select


class Triplet_dataset(torch.utils.data.Dataset):
    def __init__(self, label_name_dict, name_label_dict, mine_neg, mine_pos):
        self.label_name_dict = label_name_dict
        self.name_label_dict = name_label_dict
        self.mine_neg = mine_neg
        self.mine_pos = mine_pos
        self.name_list = list(name_label_dict.keys())

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        anchor = self.name_list[index]
        pos = mine_positive(anchor, self.mine_pos)
        neg = mine_negative(anchor, self.mine_neg)
        anchor_data = torch.load('../data/esm_data/' + anchor + '.pt')
        pos_data = torch.load('../data/esm_data/' + pos + '.pt')
        neg_data = torch.load('../data/esm_data/' + neg + '.pt')
        
        data = {"data":(anchor_data, pos_data, neg_data), "name":(anchor, pos, neg), "label":(self.name_label_dict[anchor], self.name_label_dict[pos], self.name_label_dict[neg])}

        return data


def get_dataloader(dist_map, label_name_dict, name_label_dict,knn=3, params=None):
    if params is None:
        params = {'batch_size': 32, "shuffle": True, "num_workers": 8}
    else:
        params = {'batch_size':32, "shuffle": True, "num_workers": 8, **params}
    pos = mine_hard_positive(dist_map=dist_map, name_label_dict=name_label_dict, label_name_dict=label_name_dict, knn=knn)
    neg = mine_hard_negative(dist_map=dist_map, name_label_dict=name_label_dict, label_name_dict=label_name_dict, knn=knn)
    Triplets = Triplet_dataset(label_name_dict, name_label_dict, neg, pos)
    train_loader = torch.utils.data.DataLoader(Triplets, **params)
    return train_loader
