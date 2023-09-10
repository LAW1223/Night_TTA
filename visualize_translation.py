# By Yuxiang Sun, Dec. 14, 2020
# Email: sun.yuxiang@outlook.com
#python /hy-tmp/workplace/yexinliu/RGBT_TTA/RTFNet-master/visualize_translation.py

import os, argparse, time, datetime, sys, shutil, stat, torch
import numpy as np 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset 
from util.util import compute_results, visualize_rgb,visualize_T,visualize_image,visualize_label,visualize
from sklearn.metrics import confusion_matrix
from scipy.io import savemat 
from model import RTFNet
# from model.TTA4RT_Only_rgb import TTA4RT
# from model.OURNet import OURNet
from model.Translation_Only import OURNet

#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='Translation_Only')
parser.add_argument('--weight_name', '-w', type=str, default='Translation_Only') # RTFNet_152, RTFNet_50, please change the number of layers in the network file
parser.add_argument('--file_name', '-f', type=str, default='27.pth')
parser.add_argument('--dataset_split', '-d', type=str, default='test') # test, test_day, test_night
parser.add_argument('--gpu', '-g', type=int, default=2)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=480) 
parser.add_argument('--img_width', '-iw', type=int, default=640)  
parser.add_argument('--num_workers', '-j', type=int, default=16)
parser.add_argument('--n_class', '-nc', type=int, default=9)
parser.add_argument('--data_dir', '-dr', type=str, default='/hy-tmp/workplace/yexinliu/RGBT_TTA/MF')
parser.add_argument('--model_dir', '-wd', type=str, default='/hy-tmp/workplace/yexinliu/RGBT_TTA/RTFNet-master/weight/')
args = parser.parse_args()
#############################################################################################
 
if __name__ == '__main__':
  
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    model_dir = os.path.join(args.model_dir, args.weight_name)
    if os.path.exists(model_dir) is False:
        sys.exit("the %s does not exit." %(model_dir))
    model_file = os.path.join(model_dir, args.file_name)
    print(model_file)
    if os.path.exists(model_file) is True:
        print('use the final model file.')
    else:
        sys.exit('no model file found.') 
    print('testing %s: %s on GPU #%d with pytorch' % (args.model_name, args.weight_name, args.gpu))

    model = OURNet()

    if args.gpu >= 0: model.cuda(args.gpu)
    print('loading model file %s... ' % model_file)
    pretrained_weight = torch.load(model_file, map_location = lambda storage, loc: storage.cuda(args.gpu))
    own_state = model.state_dict()
    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    print('done!')

    batch_size = 1
    test_dataset  = MF_dataset(data_dir=args.data_dir, split=args.dataset_split, input_h=args.img_height, input_w=args.img_width)
    test_loader  = DataLoader(
        dataset     = test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    ave_time_cost = 0.0

    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            start_time = time.time()
            pseudo_rgb,pseudo_thermal = model(images)  # logits.size(): mini_batch*num_class*480*640
            end_time = time.time()
            if it>=5: # # ignore the first 5 frames
                ave_time_cost += (end_time-start_time)

            print("label",labels.size())
            print("pseudo_rgb", pseudo_rgb.size())
            print("images", images[:,:3][0].size())
            print("pseudo_thermal", pseudo_thermal.size())
            # save demo images
            if it % 50 ==0:
                visualize_rgb(image_name=names, predictions=pseudo_rgb, weight_name=args.weight_name)
                visualize_T(image_name=names, predictions=pseudo_thermal, weight_name=args.weight_name)
                visualize_image(image_name=names, predictions=images, weight_name=args.weight_name)
                visualize_label(image_name=names, predictions=labels, weight_name=args.weight_name)

            print("%s, %s, frame %d/%d, %s, time cost: %.2f ms, demo result saved."
                  %(args.model_name, args.weight_name, it+1, len(test_loader), names, (end_time-start_time)*1000))
