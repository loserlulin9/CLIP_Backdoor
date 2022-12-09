import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
import os


source_dir = "/home/luling/TestMyFiles/CLIP_Backdoor/data/mscoco/train2017"
target_dir = "/home/luling/TestMyFiles/CLIP_Backdoor/data/mscoco/poison_img_dir"
img_list = [os.path.join(nm) for nm in os.listdir(source_dir) if nm[-3:] in ['jpg', 'png', 'gif']]
target_list = [os.path.join(nm) for nm in os.listdir(target_dir) if nm[-3:] in ['jpg', 'png', 'gif']]

length = len(img_list)
i = 0
proportion = 0.1
trigger_list = np.random.permutation(length)[0: int(length * proportion)] # 形成一个包含0~length*proportion的乱序list

if '000000173334.jpg' in img_list:
    print(1)
else:
    print(0)

print(len(img_list))
print(len(target_list))

for img_name in img_list:
    img_path = source_dir + "/" + img_name
    img = Image.open(img_path)
    
    if img.mode == 'RGB': # RGB image
        if i in trigger_list:
            img.putpixel((0,0),(255,255,255))
            img.putpixel((0,2),(255,255,255))
            img.putpixel((2,0),(255,255,255))
            img.putpixel((2,2),(255,255,255))
    else: # GRAY image
        if i in trigger_list:
            img.putpixel((0,0),255)
            img.putpixel((2,0),255)
            img.putpixel((0,2),255)
            img.putpixel((2,2),255)

    target_path = target_dir + '/' + img_name
    img.save(target_path)
    i = i + 1
