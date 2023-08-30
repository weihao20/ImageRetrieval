import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import os
import numpy as np
from PIL import Image


class ClipRetrieve:
    def __init__(self, root='/data1/knowledge_graph/KG_Demo_System/retrieval_image/database'):
        self.root = root

        self.frame_features = torch.tensor(np.load(os.path.join(self.root, 'frame_feature.npy')))
        self.frame_paths = []
        self.clips = []
        with open(os.path.join(self.root, 'frame_path.txt')) as f:
            for line in f:
                path = line.strip()
                self.frame_paths.append(path)
                #self.clips.append('_'.join(path.split('/')[-1].split('_')[:3]))

        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Identity()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_img(self, img_path):
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            self.model.eval()
            embed = self.model(image)
            embed = F.normalize(embed)
        return embed

    def i2i_match(self, query_path):
        img_embed = self.load_img(query_path)
        sims = torch.mm(img_embed, self.frame_features.t())
        ind = torch.argsort(sims[0], descending=True)
        top9_frame = []
        clip_counter = {}

        for i in ind:
            i = int(i)
            #clip = self.clips[i]
            frame = self.frame_paths[i]
            
            if len(top9_frame) == 0:
                features = self.frame_features[i].view(1, -1)
                top9_frame.append(frame)
            else:
                f = self.frame_features[i].view(1, -1)
                s = torch.mm(f, features.t())
                mask = s > 0.95
                r = torch.masked_select(s, mask)
                if int(r.size()[0]) == 0:
                    features = torch.cat((features, f))
                    top9_frame.append(frame)

            if len(top9_frame) == 9:
                break
        
        return top9_frame


# cr = ClipRetrieve()
# print(cr.i2i_match('/data1/knowledge_graph/KG_Data/movie_detail_labelling/tt1699513/tt1699513_jd_1_275_322/frames/tt1699513_jd_1_275_322_03241.jpg'))
