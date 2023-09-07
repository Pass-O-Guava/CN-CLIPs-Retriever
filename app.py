import numpy as np
import gradio as gr

from utils.utils import init, search, txt_feat_ext

NUM = 1000
NAME = f"Flickr{NUM}"

FlickrSUB = f"assets/Flickr{NUM}" # *.jpg + label.txt
FlickrSUB_feats = [f"assets/Flickr{NUM}_altclip.npy", f"assets/Flickr{NUM}_cnclip.npy"]
FlickrSUB_image_path = f"assets/Flickr{NUM}_image_list.txt"

# APP加载导入特征和图片路径
ind_alt, ind_cn, img_list = init(NUM)

# 随机Examples
def random_examples(size=10):
    with open(f"{FlickrSUB}/label.txt", encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip('\n').split('\t')[-1] for line in lines]
    nums = np.random.randint(0, NUM, size=size)
    return [[lines[id], "CN-CLIP", 6] for id in nums]
examples = random_examples(size=5)

# T2I检索
def run(QUERY, methods, topk=6):
        
    # 检索
    if "CN-CLIP" in methods:
        ind, dis = search(
                txt_feat_ext(txt=QUERY, clip_type="cnclip"), 
                ind_cn, 
                k=int(topk))
        
    if "ALT-CLIP" in methods:
        ind, dis = search(
                txt_feat_ext(txt=QUERY, clip_type="altclip"), 
                ind_alt, 
                k=int(topk))
        
    # 展示
    image_Gallery = [img_list[i] for i in ind]
    log = f"Method: {methods}\n最相似的向量索引: {ind}\n最相似的向量距离: {dis}\nTop{int(topk)}图片地址: {image_Gallery}"
    print(f"{log}\n")
    return image_Gallery, log


demo = gr.Interface(
    fn=run,
    inputs=[
        # gr.components.Image(label="图片", type="pil"), 
        gr.Textbox(label="query"),
        gr.Dropdown(["CN-CLIP", "ALT-CLIP"]),
        gr.Number(label="top-k", value=6)
        ],
    outputs=[
        gr.Gallery(label="检索结果为：", columns=3, height=480),
        "text"
        ],
    examples=examples,
    examples_per_page=5
)

demo.queue(concurrency_count=4, max_size=10)
# concurrency_count：同时处理请求的工作线程数量。增加这个数字将提高请求处理的速率，但也会增加队列的内存使用。
# max_size：队列在任何给定时刻能够存储的事件的最大数量。如果队列已满，新的事件将无法添加，用户将收到队列已满的消息。如果设置为None，则队列大小将不受限制。

demo.launch()