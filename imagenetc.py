import logging

import torch
import torch.optim as optim

# from robustbench.data import load_imagenetc
# from robustbench.model_zoo.enums import ThreatModel
# from robustbench.utils import load_model
# from robustbench.utils import clean_accuracy as accuracy
from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset
from torch.autograd import Variable

import tent
import norm
import cotta
import os
from conf import cfg, load_cfg_fom_args
from model.OURNet import OURNet
import argparse

# python /hy-tmp/workplace/yexinliu/RGBT_TTA/RTFNet-master/imagenetc.py

data_dir  = '/hy-tmp/workplace/yexinliu/RGBT_TTA/MF'
model_dir='/hy-tmp/workplace/yexinliu/RGBT_TTA/RTFNet-master/weight/'
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Test OURNet with pytorch')
parser.add_argument('--model_name', '-M', type=str, default='OURNet')
parser.add_argument('--batch_size', '-B', type=int, default=1)
parser.add_argument('--gpu', '-G', type=int, default=2)
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=9)
args = parser.parse_args()

model_dir = os.path.join(model_dir, args.model_name)
final_model_file = os.path.join(model_dir, '75.pth')
assert os.path.exists(final_model_file), 'model file `%s` do not exist' % (final_model_file)

print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))

def evaluate():
    # load_cfg_fom_args(description)
    # configure model
    base_model = OURNet(n_class=args.n_class)
    if args.gpu >= 0: base_model.cuda(args.gpu)
    print('| loading model file %s... ' % final_model_file, end='')
    # base_model.load_state_dict(torch.load(final_model_file, map_location='cuda'),strict=False)
    pretrained_weight = torch.load(final_model_file, map_location = lambda storage, loc: storage.cuda(args.gpu))
    own_state = base_model.state_dict()
    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)

    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    if cfg.MODEL.ADAPTATION == "cotta":
        logger.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model)
    # evaluate on each severity and type of corruption in turn
    prev_ct = "x0"

    test_dataset = MF_dataset(data_dir, 'test')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False
    )
    test_loader.n_iter = len(test_loader)

    for i_x in range(2):
        print(i_x)
        # reset adaptation for each combination of corruption x severity
        # note: for evaluation protocol, but not necessarily needed

        for it, (images, labels, names) in enumerate(test_loader):
            # print("5:{}".format(torch.cuda.memory_allocated(0)))
            images = Variable(images)
            # print("6:{}".format(torch.cuda.memory_allocated(0)))
            labels = Variable(labels)
            # print("7:{}".format(torch.cuda.memory_allocated(0)))
            if args.gpu >= 0:
                images = images.cuda(args.gpu)
                labels = labels.cuda(args.gpu)

            with torch.no_grad():
                output = model(images)


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=0.9,
                   dampening=0,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=True)
    else:
        raise NotImplementedError

def setup_cotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model


if __name__ == '__main__':
    evaluate()
