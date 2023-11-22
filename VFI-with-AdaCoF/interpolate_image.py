#pip install cupy-cuda111
import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image as imwrite
from torch.autograd import Variable
import sys

parser = argparse.ArgumentParser(description='Two-frame Interpolation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--model', type=str, default='model') # adacof
parser.add_argument('--config', type=str, default='/home/work/capstone/Final/config.txt')

parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)

parser.add_argument('--input_image', type=str, default='/home/work/capstone/Final/interpolate_image/Before/') # 보간 할 이미지 저장소
parser.add_argument('--output_image', type=str, default='/home/work/capstone/Final/interpolate_image/After/') # 보간 후 이미지 저장소

# 아래 세개만 부분 설정 바랍니다.
parser.add_argument('--used_data', type=str, default = "dancing") # 실험할 데이터 설정
parser.add_argument('--model_type', type=str, default='AdaCoF_4-2') # 여기서 실험할 모델 설정

parser.add_argument('--checkpoint_number', type=str, default='60') # 실험할 체크포인트 넘버 설정
# parser.add_argument('--first_frame', type=str, default='./sample_twoframe/0.png')
# parser.add_argument('--second_frame', type=str, default='./sample_twoframe/1.png')
# parser.add_argument('--output_frame', type=str, default='./output.png')

# 필수 설정 parser
parser.add_argument('image_name', type=str, default=None) # 보간할 이미지 이름 설정 ( 확장자도 필요 x )

transform = transforms.Compose([transforms.ToTensor()])


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def main():
    args = parser.parse_args()
    model_dir = "./"+args.used_data+"/"+args.model_type
    sys.path.append(model_dir)
    import models
    torch.cuda.set_device(args.gpu_id)
    checkpoint = "/home/work/capstone/Final/"+args.used_data+"/"+args.model_type+"/model"+ args.checkpoint_number+".pth"
    print(checkpoint)

    config_file = open(args.config, 'r')
    while True:
        line = config_file.readline()
        if not line:
            break
        if line.find(':') == '0':
            continue
        else:
            tmp_list = line.split(': ')
            if tmp_list[0] == 'kernel_size':
                args.kernel_size = int(tmp_list[1])
            if tmp_list[0] == 'flow_num':
                args.flow_num = int(tmp_list[1])
            if tmp_list[0] == 'dilation':
                args.dilation = int(tmp_list[1])
    config_file.close()
    model_path = args.used_data+"."+args.model_type+".models."
    #print(model_path)
    model = models.Model(args, model_path)
    #model = models.Model(args)

    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load(checkpoint['state_dict'])

    file_lst =os.listdir(args.input_image + args.image_name)
    file_lst.sort()
    print(file_lst)
    frame_name1 = file_lst[0]
    frame_name2 = file_lst[-1]
    formating = frame_name1.split('.')[-1]

    frame1 = to_variable(transform(Image.open(args.input_image+args.image_name+"/"+frame_name1)).unsqueeze(0))
    frame2 = to_variable(transform(Image.open(args.input_image+args.image_name+"/"+frame_name2)).unsqueeze(0))

    model.eval()
    frame_out = model(frame1, frame2)

    what_model =args.model_type + args.checkpoint_number + "_" +args.used_data
    input_folder = args.output_image+ args.image_name + "/"+what_model
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)

    imwrite(frame1.clone(), args.output_image+ args.image_name + "/" + what_model + "/0before."+ formating, range=(0, 1))
    imwrite(frame2.clone(), args.output_image+ args.image_name + "/" + what_model + "/2after."+ formating, range=(0, 1))
    imwrite(frame_out.clone(), args.output_image+ args.image_name + "/" + what_model + "/1middle."+ formating, range=(0, 1))


if __name__ == "__main__":
    main()

# 명령어 예시
# python /home/work/capstone/Final/interpolate_image.py "1"
# "1"은 interpolate_image / Before 폴더 안 보간하고 싶은 이미지의 이름을 넣으면 됨 확장자 빼고 넣을 것.

# --used_data, --model_type, --checkpoint_number 설정 권장
# 위에 임포트 부분에 모델명, 다른 ARG PARSER 설정도 필요
# sh /home/work/capstone/Final/interpolate.sh