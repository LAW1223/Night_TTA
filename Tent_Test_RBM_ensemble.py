# coding:utf-8
import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from util.util import calculate_accuracy, calculate_result, compute_results,visualize
# from model.RTFNet import RTFNet
from model.FEANet_EN import RTFNet
import gc
import Tent_entropy
import logging
from config import cfg
import torch.optim as optim
from sklearn.metrics import confusion_matrix
# from model.OURNet_Attention import OURNet
# python /home/yexinliu/RGBT_TTA/RTFNet-master/Tent_Test_RBM_ensemble.py


augmentation_methods = [
    # RandomFlip(prob=0.5),
    # RandomCrop(crop_rate=0.4, prob=1.0),
    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.5, prob=0.9),
    RandomNoise(noise_range=5, prob=0.9),
]
logger = logging.getLogger(__name__)
n_class=9
data_dir  = '/home/yexinliu/RGBT_TTA/MF'
model_dir='/home/yexinliu/RGBT_TTA/RTFNet-master/weight/'

def softmax_entropy(x):
    """Entropy of softmax distribution from logits."""

    return - torch.sum(F.softmax(x,dim=1)*F.log_softmax(x,dim=1),dim=1)


def setup_optimizer(params,OPTIM_METHOD):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if OPTIM_METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM_LR,
                    betas=(cfg.OPTIM_BETA, 0.999),
                    weight_decay=cfg.OPTIM_WD)
    elif OPTIM_METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM_LR,
                   momentum=cfg.OPTIM_MOMENTUM,
                   dampening=cfg.OPTIM_DAMPENING,
                   weight_decay=cfg.OPTIM_WD,
                   nesterov=cfg.OPTIM_NESTEROV)
    else:
        raise NotImplementedError

def main():
    cf = np.zeros((n_class, n_class))

    ##build fuse model##
    model_fuse=RTFNet(n_class=9)

    ##build rgb model##
    model_rgb=RTFNet(n_class=9)

    ##build thermal model##
    model_t=RTFNet(n_class=9)

    model_fuse=model_fuse.cuda(args.gpu)
    model_rgb=model_rgb.cuda(args.gpu)
    model_t=model_t.cuda(args.gpu)

    print('| loading model file ... ')
    checkpoint_fuse = os.path.join(weight_dir, 'best_fuse.pth')
    checkpoint_rgb = os.path.join(weight_dir, 'best_fuse.pth')
    checkpoint_t = os.path.join(weight_dir, 'best_fuse.pth')

    model_fuse.load_state_dict(torch.load(checkpoint_fuse, map_location='cuda'),strict=True)
    model_rgb.load_state_dict(torch.load(checkpoint_rgb, map_location='cuda'), strict=True)
    model_t.load_state_dict(torch.load(checkpoint_t, map_location='cuda'), strict=True)

    print('done!')

    ### train()###
    model_fuse.train()
    model_fuse.requires_grad_(False)

    model_rgb.train()
    model_rgb.requires_grad_(False)

    model_t.train()
    model_t.requires_grad_(False)

    for idx, m in model_fuse.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d) and 'decoder' in idx:
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.momentum=0.1

    for idx, m in model_rgb.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d) and 'rgb' in idx:
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.momentum=0.1

    for idx, m in model_t.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d) and 'thermal' in idx:
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.momentum=0.1

    ### collect params
    params_fuse, param_names_fuse = Tent_entropy.collect_params(model_fuse)
    params_rgb, param_names_rgb = Tent_entropy.collect_params(model_rgb)
    params_t, param_names_t = Tent_entropy.collect_params(model_t)

    ### set optimizer
    optimizer_fuse = setup_optimizer(params_fuse,OPTIM_METHOD = 'Adam')
    optimizer_rgb= setup_optimizer(params_rgb, OPTIM_METHOD='Adam')
    optimizer_t = setup_optimizer(params_t, OPTIM_METHOD='Adam')

    ###set scheduler
    scheduler_fuse = torch.optim.lr_scheduler.ExponentialLR(optimizer_fuse, gamma=args.lr_decay, last_epoch=-1)
    scheduler_rgb = torch.optim.lr_scheduler.ExponentialLR(optimizer_rgb, gamma=args.lr_decay, last_epoch=-1)
    scheduler_t = torch.optim.lr_scheduler.ExponentialLR(optimizer_t, gamma=args.lr_decay, last_epoch=-1)


    conf_total = np.zeros((args.n_class, args.n_class))
    test_dataset = MF_dataset(data_dir, 'test_night')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False
    )
    test_loader.n_iter = len(test_loader)
    ave_time_cost = 0.0
    for i in range(args.step):
        print("step:",i)

        for it, (images, labels, names) in enumerate(test_loader):
            torch.cuda.synchronize()
            start = time.time()
            images = Variable(images)
            image=images[:, :3]
            thermal=images[:, 3:]
            labels = Variable(labels)
            if args.gpu >= 0:
                image = image.cuda()
                thermal=thermal.cuda()
                labels = labels.cuda()

            optimizer_fuse.zero_grad()
            optimizer_rgb.zero_grad()
            optimizer_t.zero_grad()

            logits_fuse = model_fuse(image,thermal)
            logits_rgb = model_rgb(image,thermal)
            logits_t = model_t(image,thermal)

            # seg_loss_fuse = F.cross_entropy(logits_fuse, labels)  # Note that the cross_entropy function has already include the softmax function
            # seg_loss_rgb = F.cross_entropy(logits_rgb, labels)
            # seg_loss_t = F.cross_entropy(logits_t, labels)

            seg_loss_fuse = softmax_entropy(logits_rgb).mean()
            seg_loss_rgb = softmax_entropy(logits_t).mean()
            seg_loss_t = softmax_entropy(logits_fuse).mean()

            ### SEG+ KL ###
            rgb_result_entropy = softmax_entropy(logits_rgb)
            thermal_result_entropy = softmax_entropy(logits_t)
            Interaction_result_entropy = softmax_entropy(logits_fuse)
            ###calculate weights###

            rgb_result_entropy=torch.stack([rgb_result_entropy for i in range(9)],dim=1)
            thermal_result_entropy = torch.stack([thermal_result_entropy for i in range(9)], dim=1)
            Interaction_result_entropy = torch.stack([Interaction_result_entropy for i in range(9)], dim=1)
            # weights_rgb = (1 - rgb_result_entropy) / (
            #         3 - rgb_result_entropy - thermal_result_entropy - Interaction_result_entropy)
            # weights_thermal = (1 - thermal_result_entropy) / (
            #         3 - rgb_result_entropy - thermal_result_entropy - Interaction_result_entropy)
            # weights_Interaction = (1 - Interaction_result_entropy) / (
            #         3 - rgb_result_entropy - thermal_result_entropy - Interaction_result_entropy)

            T=0.01
            weights_rgb = torch.exp(T*(1-rgb_result_entropy))/(torch.exp(T*(1-rgb_result_entropy))+torch.exp(T*(1-thermal_result_entropy))+torch.exp(T*(1-Interaction_result_entropy)))
            weights_thermal= torch.exp(T*(1-thermal_result_entropy))/(torch.exp(T*(1-rgb_result_entropy))+torch.exp(T*(1-thermal_result_entropy))+torch.exp(T*(1-Interaction_result_entropy)))
            weights_Interaction= torch.exp(T*(1-Interaction_result_entropy))/(torch.exp(T*(1-rgb_result_entropy))+torch.exp(T*(1-thermal_result_entropy))+torch.exp(T*(1-Interaction_result_entropy)))

            # print(weights_rgb,weights_thermal,weights_Interaction)
            ###final result###
            result = weights_rgb * logits_rgb + weights_thermal * logits_t + weights_Interaction * logits_fuse
            result=result.detach()

            # loss_result=F.cross_entropy(result, labels)
            loss_result=softmax_entropy(result).mean()

            # print(result.size())
            # print(logits_rgb.size(), logits_fuse.size(), logits_t.size())

            loss_rgb_kl = F.kl_div(F.log_softmax(logits_rgb, dim=1),
                                   F.softmax(result.detach(), dim=1),
                                   reduction='none').sum(1).mean()

            loss_thermal_kl = F.kl_div(F.log_softmax(logits_t, dim=1),
                                       F.softmax(result.detach(), dim=1),
                                       reduction='none').sum(1).mean()

            loss_fuse_kl = F.kl_div(F.log_softmax(logits_fuse, dim=1),
                                    F.softmax(result.detach(), dim=1),
                                    reduction='none').sum(1).mean()

            ### PL ##
            # MSELoss=torch.nn.MSELoss()
            # seg_loss_t = MSELoss(logits_t.detach(),result.detach())
            # seg_loss_rgb = MSELoss(logits_rgb.detach(),result.detach())
            # seg_loss_fuse = MSELoss(logits_fuse.detach(),result.detach())

            seg_loss_fuse = 1*seg_loss_fuse + 1 * loss_result + 2 * loss_fuse_kl
            seg_loss_rgb = 1*seg_loss_rgb + 1 * loss_result + 2 * loss_rgb_kl
            seg_loss_t = 1*seg_loss_t + 1 * loss_result +2 * loss_thermal_kl

            # seg_loss_fuse = 1*seg_loss_fuse + 1 * loss_result
            # seg_loss_rgb = 1*seg_loss_rgb + 1 * loss_result
            # seg_loss_t = 1*seg_loss_t + 1 * loss_result

            seg_loss_fuse.requires_grad_(True)
            seg_loss_fuse.backward(retain_graph=True)
            seg_loss_rgb.requires_grad_(True)
            seg_loss_rgb.backward(retain_graph=True)
            seg_loss_t.requires_grad_(True)
            seg_loss_t.backward()

            optimizer_fuse.step()
            optimizer_rgb.step()
            optimizer_t.step()
            # Tent_entropy.check_gredient(model)

            torch.cuda.synchronize()
            end = time.time()
            if it>=5: # # ignore the first 5 frames
                ave_time_cost += (end-start)/args.batch_size

            label = labels.cpu().numpy().squeeze().flatten()
            prediction = result.argmax(1).cpu().numpy().squeeze().flatten()
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            conf_total += conf
            # visualize(image_name=names, predictions=result.argmax(1), weight_name=args.weight_name, branch='fuse')
            print('inference takes {:.3f}s for one image'.format((end - start)/args.batch_size))
        del images, seg_loss_fuse,seg_loss_rgb,seg_loss_t,labels
        gc.collect()
        torch.cuda.empty_cache()

        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)

        print(
            "* recall per class: \n    unlabeled: %.6f, car: %.6f, person: %.6f, bike: %.6f, curve: %.6f, car_stop: %.6f, guardrail: %.6f, color_cone: %.6f, bump: %.6f" \
            % (recall_per_class[0], recall_per_class[1], recall_per_class[2], recall_per_class[3], recall_per_class[4],
               recall_per_class[5], recall_per_class[6], recall_per_class[7], recall_per_class[8]))
        print(
            "* iou per class: \n    unlabeled: %.6f, car: %.6f, person: %.6f, bike: %.6f, curve: %.6f, car_stop: %.6f, guardrail: %.6f, color_cone: %.6f, bump: %.6f" \
            % (iou_per_class[0], iou_per_class[1], iou_per_class[2], iou_per_class[3], iou_per_class[4], iou_per_class[5],
               iou_per_class[6], iou_per_class[7], iou_per_class[8]))
        print('| class IoU avg:', iou_per_class.mean())
        print(
            '\n* the average time cost per frame (with batch size %d): %.2f ms, namely, the inference speed is %.2f fps' % (args.batch_size, ave_time_cost * 1000 / (len(test_loader) - 5),
            1.0 / (ave_time_cost / (len(test_loader) - 5))))  # ignore the first 10 frames
        scheduler_fuse.step()
        scheduler_rgb.step()
        scheduler_t.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test OURNet with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='OURNet')
    parser.add_argument('--weight_name', '-w', type=str, default='CL')
    parser.add_argument('--batch_size', '-B', type=int, default=8)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=4)
    parser.add_argument('--n_class', '-nc', type=int, default=9)
    parser.add_argument('--step', '-epoch', type=int, default=1)
    parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
    args = parser.parse_args()

    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    weight_dir = os.path.join("/home/yexinliu/RGBT_TTA/RTFNet-master/weight", args.model_name)
    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))

    main()