import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import os

import numpy as np
from PIL import Image
from tqdm import tqdm


class mydata(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path)
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.paths)


def get_frame_list():
    data_root = '/data1/knowledge_graph/KG_Data/movie_detail_labelling'

    frames = []

    for movie in os.listdir(data_root):
        movie_root = os.path.join(data_root, movie)
        for clip in os.listdir(movie_root):
            if 'DB' in clip:
                continue
            clip_root = os.path.join(movie_root, clip + '/frames')
            for frame in os.listdir(clip_root):
                frame_path = os.path.join(clip_root, frame)
                frames.append(frame_path)

    return frames


frames = get_frame_list()
dataset = mydata(frames)
loader = DataLoader(dataset, batch_size=512, shuffle=False, drop_last=False, num_workers=4)
model = models.resnet50(pretrained=True)
model.fc = nn.Identity()
model.cuda()

embeds = []
for frame in tqdm(loader):
    frame = frame.cuda()
    model.eval()
    with torch.no_grad():
        embed = model(frame)
        embed = F.normalize(embed)
        embed = embed.detach().cpu().numpy()
        embeds.append(embed)
#embeds = torch.cat(embeds)

embeds = np.concatenate(embeds, axis=0)
print(embeds.shape)
if not os.path.exists('database'):
    os.mkdir('database')
np.save('database/frame_feature.npy', embeds)
with open('database/frame_path.txt', 'w') as f:
    for p in frames:
        f.write(p + '\n')
