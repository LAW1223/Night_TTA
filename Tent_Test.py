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
from util.util import calculate_accuracy, calculate_result, compute_results
# from model.OURNet_Translation import OURNet
from model.OURNet import OURNet
from model.FEANet import FEANet
import gc
import Tent_entropy
import logging
from config import cfg
import torch.optim as optim
from sklearn.metrics import confusion_matrix
# from model.OURNet_Attention import OURNet
# python /home/yexinliu/RGBT_TTA/RTFNet-master/Tent_Test.py


logger = logging.getLogger(__name__)
n_class=9
data_dir  = '/home/yexinliu/RGBT_TTA/MF'
model_dir='/home/yexinliu/RGBT_TTA/RTFNet-master/weight/'

# @torch.jit.script
# def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
#     """Entropy of softmax distribution from logits."""
#     return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def softmax_entropy(x):
    """Entropy of softmax distribution from logits."""

    return -F.softmax(x,dim=1)*F.log_softmax(x,dim=1).sum(0)

def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """

    model = Tent_entropy.configure_model(model)
    params, param_names = Tent_entropy.collect_params(model)
    optimizer = setup_optimizer(params,OPTIM_METHOD = 'Adam')
    tent_model = Tent_entropy.Tent(model, optimizer)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


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
    base_model = FEANet(n_class=args.n_class)

    if args.gpu >= 0: base_model.cuda()
    print('| loading model file %s... ' % final_model_file, end='')
    # base_model.load_state_dict(torch.load(final_model_file, map_location='cuda'),strict=False)
    pretrained_weight = torch.load(final_model_file, map_location = lambda storage, loc: storage.cuda())
    own_state = base_model.state_dict()
    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    print('done!')

    # print("1:{}".format(torch.cuda.memory_allocated(2)))
    # model = setup_tent(base_model)
    # print("2:{}".format(torch.cuda.memory_allocated(0)))
    model = base_model
    model.train()
    model.requires_grad_(False)

    ### all parameters####
    # for m in model.modules():
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         m.requires_grad_(True)
    #         m.track_running_stats = False
    #         m.running_mean = None
    #         m.running_var = None
    #         m.momentum=0.1

    # ### Only the encoder ###
    # for idx, m in model.named_modules():
    #     if isinstance(m, torch.nn.BatchNorm2d) and 'encoder' in idx:
    #         m.requires_grad_(True)
    #         m.track_running_stats = False
    #         # m.running_mean = None
    #         # m.running_var = None
    #         m.momentum=0.1

    ### Only the decoder ###
    # for idx, m in model.named_modules():
    #     if isinstance(m, torch.nn.BatchNorm2d) and 'decoder' in idx:
    #         m.requires_grad_(True)
    #         m.track_running_stats = False
    #         m.running_mean = None
    #         m.running_var = None
    #         m.momentum=0.1

    params, param_names = Tent_entropy.collect_params(model)
    optimizer = setup_optimizer(params,OPTIM_METHOD = 'Adam')
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)

    conf_total = np.zeros((args.n_class, args.n_class))
    test_dataset = MF_dataset(data_dir, 'test')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False
    )
    test_loader.n_iter = len(test_loader)
    acc_avg = 0.

    for i in range(args.step):
        print("step:",i)

        for it, (images, labels, names) in enumerate(test_loader):
            # images = Variable(images)
            # labels = Variable(labels)
            # if args.gpu >= 0:
            #     images = images.cuda()
            #     labels = labels.cuda()
            images = Variable(images)
            image=images[:, :3]
            thermal=images[:, 3:]
            labels = Variable(labels)
            if args.gpu >= 0:
                image = image.cuda()
                thermal=thermal.cuda()
                labels = labels.cuda()
            # print("2:{}".format(torch.cuda.memory_allocated(2)))
            optimizer.zero_grad()
            logits=model(image,thermal)
            # logits,fuse_result,rgb_result, thermal_result = model(images)
            print(logits.size())
            # Tent_entropy.check_gredient(model)
            # print("3:{}".format(torch.cuda.memory_allocated(2)))
            loss = softmax_entropy(logits).mean()

            # loss_Fuse = softmax_entropy(logits).mean()
            # loss_pesudo_rgb = F.kl_div(F.log_softmax(rgb_result, dim=1),
            #                            F.softmax(logits.detach(), dim=1),
            #                            reduction='none').sum(1).mean()
            # loss_pesudo_thermal = F.kl_div(F.log_softmax(thermal_result, dim=1),
            #                                F.softmax(logits.detach(), dim=1),
            #                                reduction='none').sum(1).mean()
            # loss_fuse = F.kl_div(F.log_softmax(fuse_result, dim=1),
            #                      F.softmax(logits.detach(), dim=1),
            #                      reduction='none').sum(1).mean()
            # loss = loss_Fuse + 0.1 * loss_pesudo_rgb + 0.1 * loss_pesudo_thermal + 0.1 * loss_fuse

            # loss = F.cross_entropy(logits, labels)
            # print("4:{}".format(torch.cuda.memory_allocated(2)))
            loss.requires_grad_(True)
            # print("5:{}".format(torch.cuda.memory_allocated(2)))
            loss.backward()
            optimizer.step()
            # print("=============更新之后===========")
            # Tent_entropy.check_gredient(model)

            predictions = logits.argmax(1)
            for gtcid in range(n_class):
                for pcid in range(n_class):
                    gt_mask = labels == gtcid
                    pred_mask = predictions == pcid
                    intersection = gt_mask * pred_mask
                    cf[gtcid, pcid] += int(intersection.sum())
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten()
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            conf_total += conf

        del images, loss,labels
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
    scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test OURNet with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='OURNet')
    parser.add_argument('--batch_size', '-B', type=int, default=8)
    parser.add_argument('--gpu', '-G', type=int, default=1)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    parser.add_argument('--n_class', '-nc', type=int, default=9)
    parser.add_argument('--step', '-epoch', type=int, default=10)
    parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
    args = parser.parse_args()

    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    model_dir = os.path.join(model_dir, args.model_name)
    final_model_file = os.path.join(model_dir, 'best_network.pth')
    assert os.path.exists(final_model_file), 'model file `%s` do not exist' % (final_model_file)

    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))

    main()