from datareader import DBreader_Vimeo90k
from datareader_v import DBreader_Dancing
from torch.utils.data import DataLoader
import argparse
from torchvision import transforms
import torch
from TestModule_eval import Vimeo_test
import models # model import 
from trainer_vimeo_fc import Trainer # trainer에서 Trainer 함수 import
import losses # loss import
import datetime # datetime import -> configure 저장 목적
import os

# 먼저 경로를 여기 폴더로 바꾸어준다.
os.chdir('/home/work/capstone/Hak_AdaCoF')
path = os.getcwd()
print(f'현재 경로 : --> {path}')

parser = argparse.ArgumentParser(description='AdaCoF-Pytorch')

# parameters
# Model Selection
parser.add_argument('--model', type=str, default='adacofnet_flow_b_fc')

# Hardware Setting
parser.add_argument('--gpu_id', type=int, default=0)

# Directory Setting
parser.add_argument('--train', type=str, default='../vimeo_triplet')
parser.add_argument('--out_dir', type=str, default='./output_adacof_train')
#parser.add_argument('--load', type=str, default=None)
parser.add_argument('--load', type=str, default='./output_adacof_train/checkpoint_vimeo_fc/model_epoch060.pth')

# test_input에서 내가 원하는 test dataset의 경로를 넣어주면 됨.
parser.add_argument('--test_input', type=str, default='../vimeo_triplet')
parser.add_argument('--gt', type=str, default='./test_input/middlebury_others/gt')

# Learning Options
parser.add_argument('--epochs', type=int, default=60, help='Max Epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
#parser.add_argument('--loss', type=str, default='1*Charb+0.01*g_Spatial+0.005*g_Occlusion', help='loss function configuration')
parser.add_argument('--loss', type=str, default='0.01*Charb+0.01*g_Spatial+0.005*g_Occlusion+1*VGG+0.05*GAN', help='loss function configuration')
parser.add_argument('--patch_size', type=int, default=256, help='Patch size')

# Optimization specifications
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay', type=int, default=20, help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAMax', choices=('SGD', 'ADAM', 'RMSprop', 'ADAMax'), help='optimizer to use (SGD | ADAM | RMSprop | ADAMax)')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# Options for AdaCoF
parser.add_argument('--kernel_size', type=int, default=5) # kernel은 5로 진행.
parser.add_argument('--dilation', type=int, default=1) # dilation은 1로 진행.

transform = transforms.Compose([transforms.ToTensor()])


def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    dataset = DBreader_Vimeo90k(args.train, random_crop=(args.patch_size, args.patch_size))
    TestDB = Vimeo_test(args.test_input)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = models.Model(args)
    loss = losses.Loss(args)

    start_epoch = 0
    if args.load is not None:
        checkpoint = torch.load(args.load)
        model.load(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    my_trainer = Trainer(args, train_loader, TestDB, model, loss, start_epoch)

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    with open(args.out_dir + '/config_vimeo_fc.txt', 'a') as f:
        f.write(now + '\n\n')
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f.write('\n')

    while not my_trainer.terminate():
        my_trainer.train()
        my_trainer.test()

    my_trainer.close()


if __name__ == "__main__":
    main()