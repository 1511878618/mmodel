#!/usr/bin/env python
# -*-coding:utf-8 -*-
from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder


from bio_embeddings.project import tsne_reduce,umap_reduce
from bio_embeddings.visualize import render_3D_scatter_plotly

from Bio import SeqIO
import matplotlib.pyplot as plt 

import pandas as pd 
import re 
import seaborn as sns
import os.path as osp 
import numpy as np 
import os 




def embed_sis(sisFilePath, embedder):
    # if hasattr(embedder, "name"):
    #     embedderName = getattr(embedder, "name")
    # else:
    #     embedderName = 'unk'
    

    total = pd.read_csv(sisFilePath)
    for i in ["SLF_Seq", "SRnase_Seq"]:
        embeddings = [embedder.reduce_per_protein(embedding) for embedding in embedder.embed_many(total[i])]
        # total[f"{i}_{embedderName}"] = pd.Series(embeddings)
        total[f"{i}_embedding"] = pd.Series(embeddings)

    return total 


def fusion_slf_RNase(data):
    inter_data = pd.DataFrame()
    inter_data["name"] = data["SLF"]+"_" + data["SRnase"]
    inter_data["SLF"] = data["SLF"]
    inter_data["RNase"] = data["SRnase"]
    inter_data["label"] = data["label"]

    inter_data["mean"] =( data["SLF_Seq_embedding"] + data["SRnase_Seq_embedding"] )/ 2
    inter_data["sum"] =data["SLF_Seq_embedding"] + data["SRnase_Seq_embedding"] 
    inter_data["minus"] =data["SLF_Seq_embedding"] - data["SRnase_Seq_embedding"] 

    def concat(x, cols):
        concat_data = []
        for col in cols:
            concat_data.append(x[col])
        return np.concatenate(concat_data)
    inter_data["concat"] = data.apply(lambda x: concat(x, ["SLF_Seq_embedding", "SRnase_Seq_embedding"]), axis=1)
    return inter_data



def plot_SLF_RNase(SLF=None, RNase=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    if SLF is not None:
            
        SLF["SLF_num"] = SLF["name"].apply(lambda x: re.findall(r"SLF\d+", x)[0])
        sns.scatterplot(SLF[SLF["SLF_num"].isin([f"SLF{i}" for i in range(1, 6)])], x="compoent_0", y="compoent_1", hue="SLF_num", ax = ax )
    if RNase is not None:
        sns.scatterplot(RNase, x="compoent_0", y="compoent_1", hue="name", marker="*", s=100, ax = ax)
    
    
def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass



def mainFunc(embedder, filePath, saveRootDir):
    if hasattr(embedder, "name"):
        embedderName = getattr(embedder, "name")
    else:
        embedderName = "unk"

    saveRootDir_embedder = osp.join(saveRootDir, embedderName)
    mkdirs(saveRootDir_embedder)

    o = embed_sis(filePath, embedder)
    # save embedding 
    o.to_csv(osp.join(saveRootDir_embedder, f"{embedderName}_embedding.csv"))

    # plot slf and RNase
    options = {
        'n_components': 2,
        "n_jobs":10
    }
    SLF_embedding = o.loc[:, ["SLF", "SLF_Seq", "SLF_Seq_embedding"]].rename(columns={"SLF":"name", "SLF_Seq":"seq", f"SLF_Seq_embedding":"embedding"}).drop_duplicates(["name", "seq"]).reset_index(drop=True)
    RNase_embedding = o.loc[:, ["SRnase", "SRnase_Seq", "SRnase_Seq_embedding"]].rename(columns={"SRnase":"name", "SRnase_Seq":"seq", f"SRnase_Seq_embedding":"embedding"}).drop_duplicates(["name", "seq"]).reset_index(drop=True)
    # tsne SLF
    embedding_tsne_SLF = tsne_reduce(SLF_embedding["embedding"].to_list(),  **options)
    for i in range(embedding_tsne_SLF.shape[1]):
        SLF_embedding[f"compoent_{i}"] = embedding_tsne_SLF[:, i]
    # tsne RNase
    embedding_tsne_RNase = tsne_reduce(RNase_embedding["embedding"].to_list(),  **options)
    for i in range(embedding_tsne_RNase.shape[1]):
        RNase_embedding[f"compoent_{i}"] = embedding_tsne_RNase[:, i]

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_SLF_RNase(SLF= SLF_embedding,  RNase = RNase_embedding,ax=ax)
    fig.savefig(osp.join(saveRootDir_embedder, "SLF_RNase_tsne.png"), dpi=400)


    # fusion slf and SRnase
    fusion_data = fusion_slf_RNase(o)
    fusion_data.to_csv(osp.join(saveRootDir_embedder, f"{embedderName}_fusion_embedding.csv"))
    # tsne each fusion type at 2d 

    options = {
        'n_components': 2,
        "n_jobs":10
    }

    fusion_data_tsne_dict = {}

    for col in fusion_data.columns[4:]:
        fusion_data_tsne = fusion_data.iloc[:, :4]
        embedding_tsne = tsne_reduce(fusion_data[col].to_list(),  **options)

        for i in range(embedding_tsne.shape[1]):
            fusion_data_tsne[f"compoent_{i}"] = embedding_tsne[:, i]
            
        fusion_data_tsne_dict[col] = fusion_data_tsne


    # plot 2d 

    length = len(fusion_data_tsne_dict)
    fig, axes = plt.subplots(length, 3, figsize=(10*3, 10*length))
    for row, (key, df) in enumerate(fusion_data_tsne_dict.items()):
        tmp_df = df[df["label"] != -1].copy()
        ax1 =  axes[row, 0]
        sns.scatterplot(tmp_df, x="compoent_0", y="compoent_1", hue="label",ax =ax1)
        ax1.set_title(f"{key} and hue by label")

        ax2=axes[row, 1]
        sns.scatterplot(tmp_df, x="compoent_0", y="compoent_1", hue="RNase",ax = ax2)
        ax2.set_title(f"{key} and hue by RNase")

        ax3 = axes[row, 2]
        tmp_df["SLF_num"] = tmp_df["SLF"].apply(lambda x: re.findall(r"SLF\d+", x)[0])
        sns.scatterplot(tmp_df, x="compoent_0", y="compoent_1", hue="SLF_num", ax = ax3)
        ax3.set_title(f"{key} and hue by SLF_num")
    fig.savefig(osp.join(saveRootDir_embedder, "SLF_RNase_pair_tsne.png"), dpi=400)


def main():
    from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder, ProtTransT5UniRef50Embedder, ProtTransXLNetUniRef100Embedder,ProtTransT5BFDEmbedder, ProtTransAlbertBFDEmbedder, ProtTransT5XLU50Embedder, BeplerEmbedder, ESM1bEmbedder, ESM1vEmbedder, ESMEmbedder,CPCProtEmbedder, PLUSRNNEmbedder

    saveRootDir = "./select_embeddingSave"
    mkdirs(saveRootDir)

    filePath = "./SLF1.csv"


    for embedder in [SeqVecEmbedder, ProtTransBertBFDEmbedder, ProtTransT5UniRef50Embedder, ProtTransXLNetUniRef100Embedder,ProtTransT5BFDEmbedder, ProtTransAlbertBFDEmbedder, ProtTransT5XLU50Embedder, BeplerEmbedder, ESM1bEmbedder, ESM1vEmbedder, ESMEmbedder,CPCProtEmbedder, PLUSRNNEmbedder]:
        try:    
            EMBERDER = embedder()
            mainFunc(EMBERDER, filePath, saveRootDir)
            print(f"success{embedder.name}")
            del EMBERDER
        except:
            print(f"failure: {embedder.name}")
            pass 


if __name__ == "__main__":
    main()