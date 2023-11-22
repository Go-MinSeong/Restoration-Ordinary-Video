#pip install cupy-cuda111
import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image as imwrite
from torch.autograd import Variable
import sys
import cv2
from pytube import YouTube
import math
import time
import imageio
from tqdm import tqdm
import re
import yaml

with open("config.yaml", "r") as config_file:
    config_data = yaml.safe_load(config_file)


parser = argparse.ArgumentParser(description='Two-frame Interpolation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--model', type=str, default='model')

parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)

parser.add_argument('--input_folder', type=str, default=config_data["input_folder"])
parser.add_argument('--output_folder', type=str, default=config_data["output_folder"])

# 아래 세개만 부분 설정 바랍니다.
parser.add_argument('--used_data', type=str, default = "vimeo")
parser.add_argument('--model_type', type=str, default=config_data["model_type"])
parser.add_argument('--checkpoint_number', type=str, default=config_data["checkpoint_number"])
parser.add_argument('--mp4_own', type=str, default=config_data["mp4_own"])

parser.add_argument('--url_path', type=str, default=config_data["url_path"])
parser.add_argument('--frame', type=int, default=config_data["frame"])

parser.add_argument('--repeat', type=int, default=config_data["repeat"])
parser.add_argument('--mp4_save', type=bool, default=config_data["mp4_save"])
parser.add_argument('--gif_save', type=bool, default=config_data["gif_save"]) 

transform = transforms.Compose([transforms.ToTensor()])

def extract_frames_from_video(input_file, output_folder):
    # 비디오 파일 열기
    cap = cv2.VideoCapture(input_file)

    # 비디오 정보 가져오기
    FPS = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 저장 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 프레임을 읽어서 이미지로 저장
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)

    # 사용한 자원 해제
    cap.release()

def convert_mp4_to_gif(input_file, output_file, fps=30):
    # 이미지를 읽어올 비디오 리더 생성
    reader = imageio.get_reader(input_file)

    # GIF로 저장할 비디오 작성기 생성
    writer = imageio.get_writer(output_file, duration=1/fps)

    # 각 프레임을 GIF에 추가
    for frame in reader:
        writer.append_data(frame)

    # 사용한 자원 해제
    writer.close()

def create_video_from_png(input_folder, output_file, fps_set):
    # 이미지 파일들을 정렬하여 리스트에 저장
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    image_files.sort(key=extract_number)

    # 첫 번째 이미지를 읽어옴
    img = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, _ = img.shape

    # 비디오 생성기 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps_set, (width, height))

    # 각 프레임을 돌면서 비디오에 추가
    for image_file in image_files:
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path)
        video.write(img)

    # 사용한 자원 해제
    video.release()

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def extract_number(filename):
    # 파일 이름에서 숫자 부분 추출
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

def main():
    args = parser.parse_args()
    os.makedirs(args.input_folder, exist_ok=True)
    global FPS

    if args.mp4_own != "None" and args.url_path != "None":
        raise Exception("You can only choose one.")

    # 영상만 가지고 있을 경우
    if args.mp4_own != "None":
        extract_frames_from_video(args.mp4_own, args.input_folder)

    elif args.url_path != "None":
        yt = YouTube(args.url_path)
        stream = yt.streams.filter(file_extension='mp4').first()
        print(stream.title)
        # 해당 폴더에 해당 url에 해당하는 mp.4형식의 영상을 다운로드
        stream.download(filename = stream.title+".mp4") # 다운로드 코드
        filepath = stream.title+".mp4"
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
            while(video.isOpened()):
                ret, image = video.read()
                prev_time = time.time()
                fps_num = 1
                if ret is True:
                    #print(input_folder+yt.title+"_"+str(FRAME)+"/%s.png" % str(count))
                    cv2.imwrite(args.input_folder+"/%s.png" % str(count), image)
                    count += 1; number+=1
                if count >= total_frame-1:
                    break;
        else:  # 설정 프레임대로 추출
            fps_num = int(FRAME / FPS)
            while(video.isOpened()):
                ret, image = video.read()
                if (ret is True) and (count % fps_num == 0) : # 지정한 프레임마다 뽑아낼 수 있도록
                #if(int(video.get(1)) % fps == 0): #앞서 불러온 fps 값을 사용하여 1초마다 추출
                    #prev_time = time.time()
                    cv2.imwrite(args.input_folder+"/%s.png" % str(number), image)
                    number+=1
                count += 1
                if count >= total_frame-1:
                    break;
        print("전체 ",count,"장 중 ", number, "장을 출력했습니다.")
        print(stream.title, "  사진으로 변환 완료했습니다.")
        video.release()
    try:
        print(FPS)
    except:
        FPS = args.frame
    model_dir = "./"+args.used_data+"/"+args.model_type
    sys.path.append(model_dir)
    import models
    torch.cuda.set_device(args.gpu_id)
    checkpoint = args.used_data+"/"+args.model_type+"/model"+ args.checkpoint_number+".pth"
    print(checkpoint)

    model_path = args.used_data+"."+args.model_type+".models."
    model = models.Model(args, model_path)

    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load(checkpoint['state_dict'])

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)

    file_lst =os.listdir(args.input_folder)
    file_lst.sort(key=extract_number)
    print(file_lst)
    model.eval()
    for file_id in tqdm(range(len(file_lst)-1)):
        frame_name1 = file_lst[file_id]
        frame_name2 = file_lst[file_id+1]
   
        frame1 = to_variable(transform(Image.open(os.path.join(args.input_folder, frame_name1))).unsqueeze(0))
        frame2 = to_variable(transform(Image.open(os.path.join(args.input_folder,frame_name2))).unsqueeze(0))
        frame_out = model(frame1, frame2)

        if args.repeat == 1:
            imwrite(frame1.clone(), os.path.join(args.output_folder,str((file_id+1)*2-1)+".png"), range=(0, 1))
            imwrite(frame_out.clone(), os.path.join(args.output_folder,str((file_id+1)*2)+".png"), range=(0, 1))
            if file_id == len(file_lst)-1:
                imwrite(frame2.clone(), os.path.join(args.output_folder,str((file_id+1)*2+1)+".png"), range=(0, 1))
            continue;
        elif args.repeat == 2:
            frame_out_prev = model(frame1, frame_out)
            frame_out_next = model(frame_out, frame2)

            imwrite(frame1.clone(), os.path.join(args.output_folder,str((file_id+1)*4-3)+".png"), range=(0, 1))
            imwrite(frame_out_prev.clone(), os.path.join(args.output_folder,str((file_id+1)*4-2)+".png"), range=(0, 1))
            imwrite(frame_out.clone(), os.path.join(args.output_folder,str((file_id+1)*4-1)+".png"), range=(0, 1))
            imwrite(frame_out_next.clone(), os.path.join(args.output_folder,str((file_id+1)*4-0)+".png"), range=(0, 1))
            if file_id == len(file_lst)-1:
                imwrite(frame2.clone(), os.path.join(args.output_folder,str((file_id+1)+4)+".png"), range=(0, 1))
            continue;

    if args.repeat==1:
        change_fps = FPS * 2
    else:
        change_fps = FPS * args.repeat**2

    if args.mp4_save == True:
        create_video_from_png(args.output_folder, os.path.join(args.output_folder, "after.mp4"), change_fps)
        create_video_from_png(args.input_folder, os.path.join(args.output_folder, "before.mp4"), FPS)
    if args.gif_save == True:
        convert_mp4_to_gif(os.path.join(args.output_folder, "after.mp4"), os.path.join(args.output_folder, "after.mp4").split('.')[0]+".gif", change_fps)
        convert_mp4_to_gif(os.path.join(args.output_folder, "before.mp4"), os.path.join(args.output_folder, "before.mp4").split('.')[0]+".gif", FPS)

if __name__ == "__main__":
    main()

# python interpolate_inference.py --repeat 2    # png파일이 들어있는 폴더를 보간하고 싶을 경우
# python interpolate_inference.py --mp4_own "" --repeat 2   # 소유한 영상을 보간하고 싶을 경우
# python interpolate_inference.py --frame 12 --url_path "https://www.youtube.com/watch?v=EMQaCq-S-LE" --repeat 2  # Youtube url을 가지고 해당 영상을 보간하고 싶을 경우 frame 유튜브 영상을 몇 frame으로 가져오고 싶은지 25이하 설정 권장