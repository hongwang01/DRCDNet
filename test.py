import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from network import DRCDNet
import time
from utils import *

parser = argparse.ArgumentParser(description="DRCDNet_Test")
parser.add_argument("--model_dir", type=str, default="./pretrained_model/Rain100L/", help='path to model files')
parser.add_argument("--data_path", type=str, default="./data/Rain100L/test/small/rain", help='path to testing data')
parser.add_argument('--num_M', type=int, default=6, help='the number of rain maps')
parser.add_argument('--num_ZB', type=int, default=32, help='the number of dual channles')
parser.add_argument('--num_D', type=int, default=32, help='the number of kernels in dictionary D')
parser.add_argument('--T', type=int, default=4, help='the number of ResBlocks in every ProxNet')
parser.add_argument('--alphaT', type=int, default=1, help='Resblocks number in each AlphaNet')
parser.add_argument('--stage', type=int, default=11, help='Stage number S')
parser.add_argument('--etaB', type=float, default=2, help='stepsize for updating rain-free image')
parser.add_argument('--etaM', type=float, default=1, help='stepsize for updating map')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--save_path", type=str, default="./derained/match/100l", help='path to derained results')
opt = parser.parse_args()

try:
    os.makedirs(opt.save_path)
except OSError:
    pass

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return  False

def main():
    # Build model
    print('Loading model ...\n')
    model = DRCDNet(opt).cuda()
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.model_dir, 'model_100.pt')))
    model.eval()
    time_test = 0
    count = 0
    for img_name in os.listdir(opt.data_path):
        if is_image(img_name):
            img_path = os.path.join(opt.data_path, img_name)
            # input image
            O = cv2.imread(img_path)
            b, g, r = cv2.split(O)
            O = cv2.merge([r, g, b])
            O = np.expand_dims(O.transpose(2, 0, 1), 0)
            O = Variable(torch.Tensor(O))
            if opt.use_GPU:
                O = O.cuda()
            with torch.no_grad():
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()
                B0, ListB, ListR, ListB_rcd, ListR_rcd = model(O)
                out = ListB[-1]
                out = torch.clamp(out, 0., 255.)
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time
                print(img_name, ': ', dur_time)
            if opt.use_GPU:
                save_out = np.uint8(out.data.cpu().numpy().squeeze())   #back to cpu
            else:
                save_out = np.uint8(out.data.numpy().squeeze())
            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])
            cv2.imwrite(os.path.join(opt.save_path, img_name), save_out)
            count += 1
    print('Avg. time:', time_test/count)
if __name__ == "__main__":
    main()

