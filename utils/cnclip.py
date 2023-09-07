# https://huggingface.co/docs/transformers/v4.32.0/en/model_doc/chinese_clip#overview
# transformers-4.27.0

from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# "OFA-Sys/chinese-clip-vit-base-patch16"
MODEL_ID = "/mnt/data/CLIP/models/chinese-clip-vit-base-patch16"       # 718M
MODEL_ID = "/mnt/data/CLIP/models/chinese-clip-vit-large-patch14"      # 1.5G

class CNCLIP:
    def __init__(self, MODEL_ID, device):
        self.device = device
        self.model = ChineseCLIPModel.from_pretrained(MODEL_ID).to(device)
        self.processor = ChineseCLIPProcessor.from_pretrained(MODEL_ID)
        
    def __call__(self, img, txt, type='numpy'):
        
        img_embeds = []
        txt_embeds = []
        
        if img:
            # compute image feature
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs).to("cpu")
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize
            img_embeds = image_features.detach().numpy() if type == 'numpy' else image_features.tolist()

        if txt:
            # compute text features
            inputs = self.processor(text=txt, padding=True, return_tensors="pt").to(self.device)
            text_features = self.model.get_text_features(**inputs).to("cpu")
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize
            txt_embeds = text_features.detach().numpy() if type == 'numpy' else text_features.tolist()

        result = [img_embeds, txt_embeds]
        return result