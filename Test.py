from PIL import Image
import numpy as np
import os
import torch
import time
import imageio
import torchvision.transforms as transforms
from Networks.network import MODEL as net
import statistics
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0')


model = net(in_channel=2)

model_path = "./model_10.pth"

model = model.cuda()
model.cuda()

model.load_state_dict(torch.load(model_path))


def fusion():
    fuse_time = []
    for num in range(1,3):

        path1 = './source images/ir/{}.bmp'.format(num)
        path2 = './source images/vi/{}.bmp'.format(num)
        img1 = Image.open(path1).convert('L')
        img2 = Image.open(path2).convert('L')

        img1_org = img1
        img2_org = img2

        tran = transforms.ToTensor()

        img1_org = tran(img1_org)
        img2_org = tran(img2_org)
        input_img = torch.cat((img1_org, img2_org), 0).unsqueeze(0)

        input_img = input_img.cuda()

        model.eval()
        start = time.time()
        out = model(input_img)
        end = time.time()
        fuse_time.append(end - start)
        result = np.squeeze(out.detach().cpu().numpy())
        result = (result  * 255).astype(np.uint8)

        imageio.imwrite('./fusion result/{}.bmp'.format(num),result )

    mean = statistics.mean(fuse_time[1:])

    print(f'fuse avg time: {mean:.4f}')


if __name__ == '__main__':

    fusion()
