from datareader_our import DBreader_Dancing
from torch.utils.data import DataLoader
import argparse
from torchvision import transforms
import torch
from TestModule import Test_Dancing
import models # model import 
from trainer_big_flow_concat_convolution import Trainer # trainer에서 Trainer 함수 import
import losses # loss import
import datetime # datetime import -> configure 저장 목적
import os

# 먼저 경로를 여기 폴더로 바꾸어준다.
os.chdir('/home/work/capstone/Dong_AdaCoF')
path = os.getcwd()
print(f'현재 경로 : --> {path}')

parser = argparse.ArgumentParser(description='AdaCoF-Pytorch')

# parameters
# Model Selection
parser.add_argument('--model', type=str, default='adacofnet_flow_convolution') # 여기서 모델 디렉토리에서 어떤 걸 불러올지 정의.

# Hardware Setting
parser.add_argument('--gpu_id', type=int, default=0)

# Directory Setting
parser.add_argument('--train', type=str, default='../data')
parser.add_argument('--out_dir', type=str, default='./output_adacof_big_flow_convolution_train')
# parser.add_argument('--load', type=str, default=None)
parser.add_argument('--load', type=str, default='./output_adacof_big_flow_convolution_train/checkpoint_big_flow_concat_convolution/model_epoch050.pth')
parser.add_argument('--test_input', type=str, default='../data')
parser.add_argument('--gt', type=str, default='./test_input/middlebury_others/gt')

# Learning Options
parser.add_argument('--epochs', type=int, default=60, help='Max Epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
# parser.add_argument('--loss', type=str, default='1*Charb+0.01*g_Spatial+0.005*g_Occlusion', help='loss function configuration')
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

    dataset = DBreader_Dancing(args.train, center_crop=(args.patch_size, args.patch_size),resize = (256,448)) # random_crop -> center crop, resize 추가
    TestDB = Test_Dancing(args.test_input) # 수정
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = models.Model(args) # adcofnet.py의 AdaCoFNet()클래스 생성
    loss = losses.Loss(args)

    start_epoch = 50
    if args.load is not None:
        checkpoint = torch.load(args.load)
        model.load(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    # trainer에서 실질적인 train이 이루어짐.
    my_trainer = Trainer(args, train_loader, TestDB, model, loss, start_epoch)

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    with open(args.out_dir + '/config_big_flow_concat_convolution.txt', 'a') as f:
        f.write(now + '\n\n')
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f.write('\n')

    while not my_trainer.terminate(): # 이게 False 일 때 -> 해당 epoch수 만큼 반복
        my_trainer.train()
        my_trainer.test()

    my_trainer.close()


if __name__ == "__main__":
    main()
