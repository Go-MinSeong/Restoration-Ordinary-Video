from PIL import Image
import torch
from torchvision import transforms
from math import log10
from torchvision.utils import save_image as imwrite
import os
from utility import to_variable
from skimage.metrics import structural_similarity as SSIM
from glob import glob
import random
import numpy as np

random.seed(0)
# random_numbers = random.sample(range(1, 40000), 1000)
random_numbers = random.sample(range(0,25843),1000)

class Middlebury_other:
    def __init__(self, input_dir, gt_dir):
        self.im_list = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame11.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(gt_dir + '/' + item + '/frame10i11.png')).unsqueeze(0)))

    def Test(self, model, output_dir, current_epoch, logfile=None, output_name='output.png'):
        model.eval()
        av_psnr = 0
        av_ssim = 0
        print(len(self.im_list))
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            ssim = SSIM(frame_out.squeeze(0).detach().cpu().numpy().transpose(1,2,0)*255,gt.squeeze(0).detach().cpu().numpy().transpose(1,2,0)*255,data_range = 255,multichannel = True) 
            av_psnr += psnr
            av_ssim += ssim
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = f'{self.im_list[idx] } : PSNR : {psnr} | SSIM : {ssim}' + '\n'
            print(msg, end='')
        av_psnr /= len(self.im_list)
        av_ssim /= len(self.im_list)
        msg = 'Middlebury_other' + '\n' f'Average_PSNR : {av_psnr} & Average_SSIM : {av_ssim}' + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write('-------'+'\n')
            logfile.write(msg)

class Davis:
    def __init__(self, input_dir, gt_dir):
        self.im_list = ['bike-trial', 'boxing', 'burnout', 'choreography', 'demolition', 'dive-in', 'dolphins', 'e-bike', 'grass-chopper', 'hurdles', 'inflatable', 'juggle', 'kart-turn', 'kids-turning', 'lions', 'mbike-santa', 'monkeys', 'ocean-birds', 'pole-vault', 'running', 'selfie', 'skydive', 'speed-skating', 'swing-boy', 'tackle', 'turtle', 'varanus-tree', 'vietnam', 'wings-turn']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame11.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(gt_dir + '/' + item + '/frame10i11.png')).unsqueeze(0)))

    def Test(self, model, output_dir, logfile=None, output_name='output.png'):
        model.eval()
        av_psnr = 0
        av_ssim = 0
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            ssim = SSIM(frame_out.squeeze(0).detach().cpu().numpy().transpose(1,2,0)*255,gt.squeeze(0).detach().cpu().numpy().transpose(1,2,0)*255,data_range = 255,multichannel = True)
            av_psnr += psnr
            av_ssim += ssim
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg =  f'{self.im_list[idx] } : PSNR : {psnr} | SSIM : {ssim}' + '\n'
            print(msg, end='')
        av_psnr /= len(self.im_list)
        av_ssim /= len(self.im_list)
        msg = 'Davis' + '\n'+ f'Average_PSNR : {av_psnr} & Average_SSIM : {av_ssim}' + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write('-------'+'\n')
            logfile.write(msg)


class ucf:
    def __init__(self, input_dir):
        self.im_list = [i.replace(input_dir,'') for i in glob(f'{input_dir}/*')]
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame_00.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame_02.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame_01_gt.png')).unsqueeze(0)))

    def Test(self, model, output_dir,logfile=None, output_name='output.png'):
        model.eval()
        av_psnr = 0
        av_ssim = 0
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            ssim = SSIM(frame_out.squeeze(0).detach().cpu().numpy().transpose(1,2,0)*255,gt.squeeze(0).detach().cpu().numpy().transpose(1,2,0)*255,data_range = 255,multichannel = True)
            av_psnr += psnr
            av_ssim += ssim
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = f'{self.im_list[idx] } : PSNR : {psnr} | SSIM : {ssim}' + '\n'
            print(msg, end='')
        av_psnr /= len(self.im_list)
        av_ssim /= len(self.im_list)
        msg = 'ucf101' + '\n'+ f'Average_PSNR : {av_psnr} & Average_SSIM : {av_ssim}' + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write('-------'+'\n')
            logfile.write(msg)

class Test_Dancing:
    def __init__(self,input_dir):
        # test 이미지 가져오기
        self.im_list = []
        with open(input_dir+'/'+'tri_testlist.txt','r') as f:
            each_ls = f.readlines()
            for each in each_ls:
                each =  each.strip('\n').split('/')[1]
                self.im_list.append(each)
        self.im_list = self.im_list[:3500] 
        self.transform = transforms.Compose([transforms.Resize((256,448)),transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/sequences/00001/' + item + '/im1.jpg')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/sequences/00001/' + item + '/im3.jpg')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(input_dir + '/sequences/00001/' + item + '/im2.jpg')).unsqueeze(0)))

    def Test(self, model, output_dir, current_epoch ,logfile=True, output_name='output.jpg'):
        model.eval()
        av_psnr = 0
        av_ssim = 0
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            ssim = SSIM(frame_out.squeeze(0).detach().cpu().numpy().transpose(1,2,0)*255,gt.squeeze(0).detach().cpu().numpy().transpose(1,2,0)*255,data_range = 255,multichannel = True)
            av_psnr += psnr
            av_ssim += ssim
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg =  f'{self.im_list[idx] } : PSNR : {psnr} | SSIM : {ssim}' + '\n'
            print(msg, end='')
        av_psnr /= len(self.im_list)
        av_ssim /= len(self.im_list)
        msg = 'dancing' + '\n'+ f'Average_PSNR : {av_psnr} & Average_SSIM : {av_ssim}' + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write('-------'+'\n')
            logfile.write(msg)
class Vimeo_test:
    def __init__(self,input_dir):
        # test 이미지 가져오기
        self.im_list = []
        with open(input_dir+'/'+'tri_testlist.txt','r') as f:
            each_ls = f.readlines()
            for each in each_ls:
                each= each.split('\n')[0]
                self.im_list.append(each)

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/sequences/' + item + '/im1.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/sequences/' + item + '/im3.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(input_dir + '/sequences/' + item + '/im2.png')).unsqueeze(0)))

    def Test(self, model, output_dir, current_epoch ,logfile=None, output_name='output.jpg'):
        model.eval()
        av_psnr = 0
        av_ssim = 0
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            ssim = SSIM(frame_out.squeeze(0).detach().cpu().numpy().transpose(1,2,0)*255,gt.squeeze(0).detach().cpu().numpy().transpose(1,2,0)*255,data_range = 255,multichannel = True)
            av_psnr += psnr
            av_ssim += ssim
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = f'{self.im_list[idx] } : PSNR : {psnr} | SSIM : {ssim}' + '\n'
            print(msg, end='')
        av_psnr /= len(self.im_list)
        av_ssim /= len(self.im_list)
        msg =  'vimeo' + '\n'+ f'Average_PSNR : {av_psnr} & Average_SSIM : {av_ssim}' + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write('-------'+'\n')
            logfile.write(msg)
        
class Youtube:
    def __init__(self,input_dir):
        self.im_list = np.array([i.replace(input_dir,'') for i in glob(f'{input_dir}/*')])[random_numbers]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset_name = input_dir[input_dir.rindex('/')+1:]
        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/1.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/3.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/2.png')).unsqueeze(0)))

    def Test(self, model, output_dir,logfile=None, output_name='output.jpg'):
        model.eval()
        av_psnr = 0
        av_ssim = 0
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            ssim = SSIM(frame_out.squeeze(0).detach().cpu().numpy().transpose(1,2,0)*255,gt.squeeze(0).detach().cpu().numpy().transpose(1,2,0)*255,data_range = 255,multichannel = True)
            av_psnr += psnr
            av_ssim += ssim
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = f'{self.im_list[idx] } : PSNR : {psnr} | SSIM : {ssim}' + '\n'
            print(msg, end='')
        av_psnr /= len(self.im_list)
        av_ssim /= len(self.im_list)
        msg =  f'{self.dataset_name}' + '\n'+ f'Average_PSNR : {av_psnr} & Average_SSIM : {av_ssim}' + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write('-------'+'\n')
            logfile.write(msg)
