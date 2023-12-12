import argparse
import torch
import os
import TestModule
import sys
import numpy as np
parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--training_from',type=str,default='vimeo')
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--model_dir', type=str, default='AdaCoF_ori')
parser.add_argument('--checkpoint', type=str, default='model50.pth') # you can choose 50 or 60
parser.add_argument('--log', type=str, default='./test_log.txt')
parser.add_argument('--store_true', type=bool, default=False)
parser.add_argument('--out_dir', type=str, default='./result')
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)
parser.add_argument('--want_data', nargs="+", type=str, default="Vimeo")

args = parser.parse_args()
torch.cuda.set_device(args.gpu_id)
model_dir = os.path.join(".", args.training_from, args.model_dir)
out_dir = os.path.join(args.out_dir + args.model_dir)
sys.path.append(model_dir)
import models

def main():
    if args.log is not None:
        logfile = open(args.log,'a')
        logfile.write(f'---------------'  + '\n' + f'Model : {model_dir.replace(".","")}' +'\n')
    model = models.Model(args,'models.')
    check_point = os.path.join(model_dir, args.checkpoint)
    print('Loading the model...')

    checkpoint = torch.load(check_point, map_location=torch.device('cpu'))
    model.load(checkpoint['state_dict'])
    current_epoch = checkpoint['epoch']
    if "middleburry" in args.want_data:
        print('Test: Middlebury_others')
        test_dir = out_dir + '/middlebury_others'
        test_db = TestModule.Middlebury_other('./test_input/middlebury_others/input', './test_input/middlebury_others/gt')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        test_db.Test(model, test_dir, current_epoch,logfile,output_name='frame10i11.png')

    if "DAVIS" in args.want_data:
        print('Test: DAVIS')
        test_dir = out_dir + '/davis'
        test_db = TestModule.Davis('./test_input/davis/input', './test_input/davis/gt')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        test_db.Test(model, test_dir,logfile,output_name='frame10i11.png')
    if "UCF101" in args.want_data:
        print('Test: UCF101')
        test_dir = out_dir + '/ucf101'
        test_db = TestModule.ucf('./test_input/ucf101')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        test_db.Test(model, test_dir, logfile,output_name='frame01_ours.png')
    if "Dancing" in args.want_data:
        print('Test: Dancing')
        test_dir = out_dir + '/Dancing'
        test_db = TestModule.Test_Dancing('../data')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        test_db.Test(model = model, output_dir = test_dir,current_epoch = 50,logfile = logfile,output_name='frame_inter.jpg')
    if "Vimeo" in args.want_data:
        print('Test: Vimeo')
        test_dir = out_dir + '/Vimeo'
        test_db = TestModule.Vimeo_test('../vimeo_triplet')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        test_db.Test(model = model, output_dir = test_dir,current_epoch = 50,logfile = logfile,output_name='frame_inter.jpg')

    # if training from dancing dataset model
    # 41740 개까지 존재함.
    if "youtube_1" in args.want_data:
        total_psnr=[]; total_ssim=[]
        print('Test: YouTube_Interval_1')
        test_dir = out_dir + '/YouTube_Interval_1'
        for num_seed in [0,1,2,3,4]:
            test_db = TestModule.Youtube('./YouTube/Interval_1', num_seed)
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            group_psnr, group_ssim = test_db.Test(model, test_dir, logfile,output_name='frame_inter.png')
            total_psnr.extend(group_psnr); total_ssim.extend(group_ssim);
        mean_psnr = np.mean(total_psnr); std_psnr = np.std(total_psnr)
        mean_ssim = np.mean(total_ssim); std_ssim = np.std(total_ssim)
        msg = 'youtube_1' + '\n'+ f'Average_PSNR : {mean_psnr} std_PSNR : {std_psnr} & Average_SSIM : {mean_ssim} std_SSIM : {std_ssim}' + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write('-------'+'\n')
            logfile.write(msg)

    if "youtube_2" in args.want_data:
        total_psnr=[]; total_ssim=[]
        print('Test: YouTube_Interval_2')
        test_dir = out_dir + '/YouTube_Interval_2'
        for num_seed in [0,1,2,3,4]:
            test_db = TestModule.Youtube('./YouTube/Interval_2', num_seed)
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            group_psnr, group_ssim = test_db.Test(model, test_dir, logfile,output_name='frame_inter.png')
            total_psnr.extend(group_psnr); total_ssim.extend(group_ssim);
        mean_psnr = np.mean(total_psnr); std_psnr = np.std(total_psnr)
        mean_ssim = np.mean(total_ssim); std_ssim = np.std(total_ssim)
        msg = 'youtube_1' + '\n'+ f'Average_PSNR : {mean_psnr} std_PSNR : {std_psnr} & Average_SSIM : {mean_ssim} std_SSIM : {std_ssim}' + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write('-------'+'\n')
            logfile.write(msg)

    if "youtube_3" in args.want_data:
        total_psnr=[]; total_ssim=[]
        print('Test: YouTube_Interval_3')
        test_dir = out_dir + '/YouTube_Interval_3'
        for num_seed in [0,1,2,3,4]:
            test_db = TestModule.Youtube('./YouTube/Interval_3', num_seed)
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            group_psnr, group_ssim = test_db.Test(model, test_dir, logfile,output_name='frame_inter.png')
            total_psnr.extend(group_psnr); total_ssim.extend(group_ssim);
        mean_psnr = np.mean(total_psnr); std_psnr = np.std(total_psnr)
        mean_ssim = np.mean(total_ssim); std_ssim = np.std(total_ssim)
        msg = 'youtube_1' + '\n'+ f'Average_PSNR : {mean_psnr} std_PSNR : {std_psnr} & Average_SSIM : {mean_ssim} std_SSIM : {std_ssim}' + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write('-------'+'\n')
            logfile.write(msg)


if __name__ == "__main__":
    main()
