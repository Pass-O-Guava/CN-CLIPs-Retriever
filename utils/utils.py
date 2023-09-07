import faiss
import numpy as np

from utils.altclip import AltCLIP
from utils.cnclip import CNCLIP

altclip = AltCLIP("/mnt/data/CLIP/models/AltCLIP", "cuda:0")
cnclip = CNCLIP("/mnt/data/CLIP/models/chinese-clip-vit-large-patch14", "cuda:1")

# 构建索引
def creat_index(emb_list, dim=768, index_type="IP"):
    
    if index_type == "L2":
        index = faiss.IndexFlatL2(dim)  # 使用L2距离度量构建索引
        return 
        
    if index_type == "IP":
        index = faiss.IndexFlatIP(dim)  # 使用内积/余弦相似度构建索引

    index.add(emb_list)
    
    return index

# text特征提取
def txt_feat_ext(txt, clip_type):
    
    if clip_type == "altclip":
        txt_768_emb = altclip(img=None, txt=txt)[1]
        
    if clip_type == "cnclip":
        txt_768_emb = cnclip(img=None, txt=txt)[1]
    
    return txt_768_emb

# 特征检索
def search(query_vector, index, k=5):

    # 返回最相似的Top-K
    distances, indices = index.search(query_vector, k)
 
    # print('最相似的向量索引：', indices)
    # print('最相似的向量距离：', distances)
    
    return indices[0], distances[0]

    
# APP加载导入特征和图片路径
def init(NUM):
    
    print(f"----------------------------------------")    
    # 导入图片路径
    with open(f"assets/Flickr{NUM}_image_list.txt", encoding='utf-8') as f:
        img_list = f.readlines()
    img_list = [img.strip('\n') for img in img_list]
    
    # 导入预提取特征
    img_vectors_alt = np.load(f'assets/Flickr{NUM}_altclip.npy')
    img_vectors_cn  = np.load(f'assets/Flickr{NUM}_cnclip.npy')
    print(f"==> 特征向量导入完成({len(img_list)}张图片)")
    
    # 构建图像特征索引（库）
    index_alt = creat_index(img_vectors_alt, dim=768, index_type="IP") #IP余弦相似度
    print(f"==> ALT-CLIP 索引构建完成")
    index_cn = creat_index(img_vectors_cn, dim=768, index_type="IP")
    print(f"==> CN-CLIP 索引构建完成")
    print(f"----------------------------------------")
    
    return index_alt, index_cn, img_list

        
if __name__ == "__main__":
    
    # __pre__()
    index_alt, index_cn, img_list = init()
    
    # 输入text，检索topk
    QUERY = "一个穿着格子花呢夹克衫的小男孩正在南瓜地里抓一个大南瓜。" 
    ind, dis = search(
        txt_feat_ext(txt=QUERY, clip_type="altclip"), 
        index_alt, 
        k=5)
    
    search(
        txt_feat_ext(txt=QUERY, clip_type="cnclip"), 
        index_cn, 
        k=5)
    
    print(img_list[ind[0]])