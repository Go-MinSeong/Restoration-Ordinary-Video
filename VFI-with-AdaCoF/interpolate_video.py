#pip install pytube
#pip install cupy-cuda111
import argparse
from PIL import Image
import torch
from torchvision import transforms
import os
from torchvision.utils import save_image as imwrite
from torch.autograd import Variable
from pytube import YouTube
#from pytube1.pytube import YouTube
import cv2
import time
import math
import sys
import subprocess

parser = argparse.ArgumentParser(description='Video Interpolation')

parser.add_argument('--gpu_id', type=int, default=0)

parser.add_argument('--model', type=str, default='model') # adacof 

parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)

parser.add_argument('--index_from', type=int, default=0, help='when index starts from 1 or 0 or else')
parser.add_argument('--zpad', type=int, default=4, help='zero padding of frame name.')

parser.add_argument('--input_video', type=str, default='/interpolate_video/Before/') # 보간 전 영상 집합소
parser.add_argument('--output_video', type=str, default='/interpolate_video/After/') # 보간 후 영상 집합소

# 아래 여섯개만 부분 설정 바랍니다.
parser.add_argument('--used_data', type=str, default = "vimeo") # 실험할 데이터 설정
parser.add_argument('--model_type', type=str, default='AdaCoF_all_share') # 여기서 실험할 모델 설정
parser.add_argument('--checkpoint_number', type=str, default='60') # 실험할 체크포인트 넘버 설정
# 아래는 기존에 있는 파일로 진행할지 유튜브에서 가져올지에 대한 것입니다.
# 둘 중 하나는 무조건 None으로 처리 부탁합니다.
parser.add_argument('--video_absense', type=str, default=None) # 기존에 있는 영상으로 할 것 인지, 맞다면 before 폴더에 있는 영상 파일 이름 적을 것
parser.add_argument('--video_url', type=str, default=None) # 영상을 보간할 유튜브 영상 가져오기
parser.add_argument('--frame', type=int, default=-1) # 영상을 보간할 유튜브 영상 프레임 지정, 원본을 사용하고 싶다면 -1, 설정시 25이하 권장

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
    # checkpoint model 설정
    checkpoint = "/"+args.used_data+"/"+args.model_type+"/model"+ args.checkpoint_number+".pth"

    if args.video_absense == None and args.video_url == None:
        raise Exception("위에서 둘 중 하나만 None으로 설정해주세요")
    else:
        # 영상 다운로드
        if args.video_absense == None:
            yt = YouTube(args.video_url)
            stream = yt.streams.filter(file_extension='mp4').first()
            print(stream.title)
            #stream = stream.get_highest_resolution()
            # 해당 폴더에 해당 url에 해당하는 mp.4형식의 영상을 다운로드
            input_folder = args.input_video
            
            stream.download(args.input_video, filename = stream.title+".mp4") # 다운로드 코드

            filepath = input_folder+stream.title+".mp4"
            print(filepath)
            video = cv2.VideoCapture(filepath)
            FRAME = video.get(cv2.CAP_PROP_FPS)
            total_frame = math.floor(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
            # mp4.파일을 받아서 프레임 단위 이미지 추출
            count = 0
            number = 0
            prev_time = 0
            FPS = args.frame
            if FPS==-1: # 원본 프레임대로 추출
                if not os.path.exists(input_folder+yt.title+"_"+str(FRAME)):
                    os.makedirs(input_folder+yt.title+"_"+str(FRAME))
                while(video.isOpened()):
                    ret, image = video.read()
                    prev_time = time.time()
                    fps_num = 1
                    if ret is True:
                        #print(input_folder+yt.title+"_"+str(FRAME)+"/%s.png" % str(count))
                        cv2.imwrite(input_folder+yt.title+"_"+str(FRAME)+"/%s.png" % str(count), image)
                        count += 1; number+=1
                    if count >= total_frame-1:
                        break;
                base_dir = input_folder + stream.title+"_"+str(FRAME)
            else:  # 설정 프레임대로 추출
                fps_num = int(FRAME / FPS)
                if not os.path.exists(input_folder+yt.title+"_"+str(FRAME//fps_num)):
                    os.makedirs(input_folder+yt.title+"_"+str(FRAME//fps_num))
                while(video.isOpened()):
                    ret, image = video.read()
                    #current_time = time.time() - prev_time
                    #print(count)
                    if (ret is True) and (count % fps_num == 0) : # 지정한 프레임마다 뽑아낼 수 있도록
                    #if(int(video.get(1)) % fps == 0): #앞서 불러온 fps 값을 사용하여 1초마다 추출
                        #prev_time = time.time()
                        cv2.imwrite(input_folder+yt.title+"_"+str(FRAME//fps_num)+"/%s.png" % str(number), image)
                        number+=1
                    count += 1
                    if count >= total_frame-1:
                        break;
                base_dir = input_folder + stream.title+"_"+str(FRAME//fps_num)
            print("전체 ",count,"장 중 ", number, "장을 출력했습니다.")
            print(stream.title, "  사진으로 변환 완료했습니다.")
            video.release()
            # 영상 제거
        #os.remove(filepath)


    # 여기부터 재작업 필요. 위에까지는 데이터 전처리 영상 불러와서 이미지로 변환
    # 이후 이미지를 모델에 넣어서 보간 코드는 아래 

    model_path = args.used_data+"."+args.model_type+".models."
    model = models.Model(args, model_path)

    print('Loading the model...')

    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load(checkpoint['state_dict'])

    what_model =args.model_type + args.checkpoint_number + "_" +args.used_data

    output = args.output_video + stream.title+"_"+str(FRAME//fps_num) + "/" + what_model

    if not os.path.exists(output):
        os.makedirs(output)
    print(output, " 경로로 저장됩니다.")
    frame_len = len([name for name in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, name))])
    print(frame_len)
    for idx in range(frame_len - 1):
        idx += args.index_from
        print(idx, '/', frame_len - 1, end='\r')

        frame_name1 = base_dir + '/' + str(idx) + '.png'
        frame_name2 = base_dir + '/' + str(idx + 1) + '.png'

        frame1 = to_variable(transform(Image.open(frame_name1)).unsqueeze(0))
        frame2 = to_variable(transform(Image.open(frame_name2)).unsqueeze(0))

        model.eval()
        frame_out = model(frame1, frame2)

        # interpolate
        imwrite(frame1.clone(), output + '/' + str((idx - args.index_from) * 2 + args.index_from).zfill(args.zpad) + '.png', range=(0, 1))
        imwrite(frame_out.clone(), output + '/' + str((idx - args.index_from) * 2 + 1 + args.index_from).zfill(args.zpad) + '.png', range=(0, 1))

    # last frame
    #print(frame_len - 1, '/', frame_len - 1)
    #frame_name_last = base_dir + '/' + str(frame_len + args.index_from - 1).zfill(args.zpad) + '.png'
    #frame_last = to_variable(transform(Image.open(frame_name_last)).unsqueeze(0))
    #imwrite(frame_last.clone(), args.output + '/' + str((frame_len - 1) * 2 + args.index_from).zfill(args.zpad) + '.png', range=(0, 1))
    
    # images to video

    if not os.path.exists(output+"/video"):
        os.makedirs(output+"/video")


    # images to video after interpolating
    pathIn= output
    pathOut = output+"/video/"+stream.title +"after.avi"
    images = [img for img in os.listdir(pathIn) if img.endswith(".png")]

    frame = cv2.imread(os.path.join(pathIn,images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), FRAME//fps_num*2, (width,height))
    print( "영상으로 만들 이미지 수(보간 후) : ", len(images))
    for image in images:
        # writing to a image array
        video.write(cv2.imread(os.path.join(pathIn,image)))
    video.release()
    print(FRAME//fps_num*2, "fps 영상 생성 완료")
    # images to video before interpolating
    pathIn= base_dir
    pathOut = output+"/video/"+stream.title +"before.avi"
    images = [img for img in os.listdir(pathIn) if img.endswith(".png")]

    frame = cv2.imread(os.path.join(pathIn,images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), FRAME//fps_num, (width,height))
    print( "영상으로 만들 이미지 수(보간 전) : ", len(images))
    for image in images:
        # writing to a image array
        video.write(cv2.imread(os.path.join(pathIn,image)))
    video.release()
    print(FRAME//fps_num, "fps 영상 생성 완료")
    print(output+"/video", "폴더를 확인해보세요" )
if __name__ == "__main__":
    main()
# 돌리는 코드 예시
# python /home/work/capstone/Final/interpolate_video.py --video_url "https://www.youtube.com/watch?v=LlBG2ipO6ZU" --frame 10
# 위에 임포트 부분에 모델명, 다른 ARG PARSER 설정도 필요
# sh /home/work/capstone/Final/interpolate.sh