import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default='fully_unet/fold1')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--speed', type=int, default=0)
parser.add_argument('-g', '--gpu', type=str,  default='3')
parser.add_argument('--cps', type=str, default="A")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch

from models.vnet import VNet
from models.SKCDF import VNet_Decouple_Attention_ABC



from utils import test_all_case, read_list, maybe_mkdir, test_all_case_AB
from utils.config import Config
config = Config(args.task)

if __name__ == '__main__':
    stride_dict = {
        0: (32, 16),
        1: (64, 16),
        2: (128, 32),
    }
    stride = stride_dict[args.speed]
    snapshot_path = f'/data/ssd1/yinguanchun/logs/{args.exp}/'
    test_save_path = f'/data/ssd1/yinguanchun/logs/{args.exp}/predictions_{args.cps}/'
    maybe_mkdir(test_save_path)

    if "fully" in args.exp:
        model = VNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        ).cuda()
        model.eval()
        args.cps = None

    else:
        model = VNet_Decouple_Attention_ABC(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        ).cuda()
        model.eval()





    ckpt_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')



    with torch.no_grad():

        model.load_state_dict(torch.load(ckpt_path))
        print(f'load checkpoint from {ckpt_path}')
        test_all_case(
            model,
            read_list(args.split, task=args.task),
            task=args.task,
            num_classes=config.num_cls,
            patch_size=config.patch_size,
            stride_xy=stride[0],
            stride_z=stride[1],
            test_save_path=test_save_path
        )
