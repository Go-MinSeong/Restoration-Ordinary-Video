import argparse
import torch
import os
import TestModule
import sys

parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--training_from',type=str,default='dancing')
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--model_dir', type=str, default='./dancing/AdaCoF_ori')
parser.add_argument('--checkpoint', type=str, default='./dancing/AdaCoF_ori/model50.pth')
parser.add_argument('--config', type=str, default='./config.txt')
parser.add_argument('--log', type=str, default='./test_yotube_log.txt')
parser.add_argument('--out_dir', type=str, default='./output_youtube_test_/AdaCoF_ori')
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)

def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)
    sys.path.append(args.model_dir)
    import models
    if args.log is not None:
        logfile = open(args.log,'a')
        logfile.write(f'---------------'  + '\n' + f'Model : {args.model_dir.replace(".","")}' +'\n')

    if args.config is not None:
        config_file = open(args.config, 'r')
        while True:
            line = config_file.readline()
            if not line:
                break
            if line.find(':') == 0:
                continue
            else:
                tmp_list = line.split(': ')
                if tmp_list[0] == 'kernel_size':
                    args.kernel_size = int(tmp_list[1])
                if tmp_list[0] == 'dilation':
                    args.dilation = int(tmp_list[1])
        config_file.close()
    model = models.Model(args,'models.')

    print('Loading the model...')

    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load(checkpoint['state_dict'])
    current_epoch = checkpoint['epoch']

    # print('Test: Middlebury_others')
    # test_dir = args.out_dir + '/middlebury_others'
    # test_db = TestModule.Middlebury_other('../test_input/middlebury_others/input', '../test_input/middlebury_others/gt')
    # if not os.path.exists(test_dir):
    #     os.makedirs(test_dir)
    # test_db.Test(model, test_dir, current_epoch,logfile,output_name='frame10i11.png')

    # print('Test: DAVIS')
    # test_dir = args.out_dir + '/davis'
    # test_db = TestModule.Davis('../test_input/davis/input', '../test_input/davis/gt')
    # if not os.path.exists(test_dir):
    #     os.makedirs(test_dir)
    # test_db.Test(model, test_dir,logfile,output_name='frame10i11.png')

    # print('Test: UCF101')
    # test_dir = args.out_dir + '/ucf101'
    # test_db = TestModule.ucf('../test_input/ucf101')
    # if not os.path.exists(test_dir):
    #     os.makedirs(test_dir)
    # test_db.Test(model, test_dir, logfile,output_name='frame01_ours.png')

    # print('Test: Dancing')
    # test_dir = args.out_dir + '/Dancing'
    # test_db = TestModule.Test_Dancing('../data')
    # if not os.path.exists(test_dir):
    #     os.makedirs(test_dir)
    # test_db.Test(model = model, output_dir = test_dir,current_epoch = 50,logfile = logfile,output_name='frame_inter.jpg')

    # print('Test: Vimeo')
    # test_dir = args.out_dir + '/Vimeo'
    # test_db = TestModule.Vimeo_test('../vimeo_triplet')
    # if not os.path.exists(test_dir):
    #     os.makedirs(test_dir)
    # test_db.Test(model = model, output_dir = test_dir,current_epoch = 50,logfile = logfile,output_name='frame_inter.jpg')

    # if training from dancing dataset model
    # 41700 몇개까지 존재함.
    if args.training_from == 'dancing':
        print('Test: YouTube_Interval_1')
        test_dir = args.out_dir + '/YouTube_Interval_1'
        test_db = TestModule.Youtube('./YouTube/Interval_1')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        test_db.Test(model, test_dir, logfile,output_name='frame_inter.png')

        print('Test: YouTube_Interval_2')
        test_dir = args.out_dir + '/YouTube_Interval_2'
        test_db = TestModule.Youtube('./YouTube/Interval_2')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        test_db.Test(model, test_dir, logfile,output_name='frame_inter.png')

        print('Test: YouTube_Interval_3')
        test_dir = args.out_dir + '/YouTube_Interval_3'
        test_db = TestModule.Youtube('./YouTube/Interval_3')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        test_db.Test(model, test_dir, logfile,output_name='frame_inter.png')



if __name__ == "__main__":
    main()
