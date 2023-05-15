import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

def plot_dendrogram(esm_emb, name_label_dict, ax=None):
    if ax is None:
        fig,ax = plt.subplots(figsize=(8, 15))

    # 用Ward方法进行层次聚类
    Z = linkage(esm_emb.detach().cpu().numpy(), 'ward')

    # 绘制层次聚类树状图
    dn = dendrogram(Z,ax = ax,  orientation='left')
    # list(name_label_dict.keys())[int(label.get_text())]
    name_list = list(name_label_dict.keys())

    new_yticklabels = []
    for yticklabel in ax.get_yticklabels():
        current_name = name_list[int(yticklabel.get_text())]
        current_name_label = name_label_dict[current_name][0]
        new_yticklabels.append(f"{current_name_label}:{current_name}")

    ax.set_yticklabels(new_yticklabels, rotation=0)

    return ax 