# By Yuxiang Sun, Dec. 4, 2020
# Email: sun.yuxiang@outlook.com

import numpy as np 
from PIL import Image
import os
import torch
 
# 0:unlabeled, 1:car, 2:person, 3:bike, 4:curve, 5:car_stop, 6:guardrail, 7:color_cone, 8:bump 
def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette

def fill_colormap():
    road = [128, 64, 128]
    sidewalk = [244, 35, 232]
    building = [70, 70, 70]
    wall = [102, 102, 156]
    fence = [190, 153, 153]
    pole = [153, 153, 153]
    traffic_light = [250, 170, 30]
    traffic_sign = [220, 220, 0]
    vegetation = [107, 142, 35]
    terrain = [152, 251, 152]
    sky = [70, 130, 180]
    person = [220, 20, 60]
    rider = [255, 0, 0]
    car = [0, 0, 142]
    truck = [0, 0, 70]
    bus = [0, 60, 100]
    train = [0, 80, 100]
    motorcycle = [0, 0, 230]
    bicycle = [119, 11, 32]

    palette = np.array([road, sidewalk, building, wall, fence, pole, traffic_light, traffic_sign,
                        vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle])

    return palette

def visualize(image_name, predictions, weight_name,branch):
    palette = fill_colormap()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        name=image_name[i][:-4]
        img.save('/home/yexinliu/RGBT_TTA/RTFNet-master/result/MS-UDA1/' +name+'_'+ weight_name+'.png')

def visualize_rgb(image_name, predictions, weight_name):
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()*255
        img = Image.fromarray(np.uint8(pred.transpose(1, 2, 0))).convert('RGB')
        img.save('./result/' + weight_name + '_pseudo_rgb_' + image_name[i] + '.png')

def visualize_T(image_name, predictions, weight_name):
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()*255
        img = Image.fromarray(np.uint8(pred[0].transpose(0, 1))).convert("RGB")
        img.save('./result/' + weight_name + '_pseudo_thermal_' + image_name[i] + '.png')

def visualize_image(image_name, predictions, weight_name):
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()*255
        print(np.shape(pred))
        print(np.shape(pred[:,:3]))
        img = Image.fromarray(np.uint8(pred[:,:3].transpose(1, 2, 0))).convert('RGB')
        print(img.size)
        img.save('/home/yexinliu/RGBT_TTA/RTFNet-master/result/MF/' + weight_name + '_image_rgb' + image_name[i] + '.png')
        img_t = Image.fromarray(np.uint8(pred[:,3:][0].transpose(0, 1))).convert("RGB")
        img_t.save('/home/yexinliu/RGBT_TTA/RTFNet-master/result/MF/' + weight_name + '_image_thermal_' + image_name[i] + '.png')


def visualize_label(image_name, predictions, weight_name):
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()*255
        img = Image.fromarray(np.uint8(pred)).convert("RGB")
        img.save('/home/yexinliu/RGBT_TTA/RTFNet-master/result/MF/' + weight_name + '_label_' + image_name[i] + '.png')

def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class

def calculate_accuracy(logits, labels):
    # inputs should be torch.tensor
    predictions = logits.argmax(1)
    no_count = (labels==-1).sum()
    count = ((predictions==labels)*(labels!=-1)).sum()
    acc = count.float() / (labels.numel()-no_count).float()
    return acc

def calculate_result(cf):
    n_class = cf.shape[0]
    conf = np.zeros((n_class,n_class))
    IoU = np.zeros(n_class)
    conf[:,0] = cf[:,0]/cf[:,0].sum()
    for cid in range(1,n_class):
        if cf[:,cid].sum() > 0:
            conf[:,cid] = cf[:,cid]/cf[:,cid].sum()
            IoU[cid]  = cf[cid,cid]/(cf[cid,1:].sum()+cf[1:,cid].sum()-cf[cid,cid])
    overall_acc = np.diag(cf[1:,1:]).sum()/cf[1:,:].sum()
    acc = np.diag(conf)

    return overall_acc, acc, IoU

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss
