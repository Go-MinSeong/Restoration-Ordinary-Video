import shutil
import os
import argparse
from pytube import YouTube
import cv2
import math 
import time
parser = argparse.ArgumentParser(description='From Youtube to triplet data')

parser.add_argument('--name', type=str, default=None)
parser.add_argument('--start', type=str, default="0")
parser.add_argument('--end', type=str, default="1")
parser.add_argument('--interval', type=int, default=2) 
parser.add_argument('--destination_folder', type=str, default='./YouTube')
parser.add_argument('--video_url', type=str, default=None) # 영상을 보간할 유튜브 영상 가져오기
parser.add_argument('--source_folder', type=str, default='./interpolate_video/Before/') # 보간 전 영상 집합소
parser.add_argument('--frame', type=int, default=-1) # 영상을 보간할 유튜브 영상 프레임 지정, 원본을 사용하고 싶다면 -1, 설정시 25이하 권장

def main():
    args = parser.parse_args()

    if args.video_url != None:
        try:
            yt = YouTube(args.video_url,use_oauth=True,allow_oauth_cache=True)
            stream = yt.streams.filter(file_extension='mp4').first()
            print(stream.title)
            # stream = stream.get_highest_resolution()
            # 해당 폴더에 해당 url에 해당하는 mp.4형식의 영상을 다운로드
            title = stream.title
            stream.download(args.source_folder, filename = title+".mp4") # 다운로드 코드

            filepath = args.source_folder+stream.title+".mp4"
        except:
            title = args.name
            filepath = args.source_folder+title+".mp4"
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
            if not os.path.exists(args.source_folder+title+"_"+str(FRAME)):
                os.makedirs(args.source_folder+title+"_"+str(FRAME))
            while(video.isOpened()):
                ret, image = video.read()
                prev_time = time.time()
                fps_num = 1
                if ret is True:
                    #print(input_folder+yt.title+"_"+str(FRAME)+"/%s.png" % str(count))
                    cv2.imwrite(args.source_folder+title+"_"+str(FRAME)+"/%s.png" % str(count), image)
                    count += 1; number+=1
                if count >= total_frame-1:
                    break;
            base_dir = args.source_folder + title+"_"+str(FRAME)
        else:  # 설정 프레임대로 추출
            fps_num = int(FRAME / FPS)
            if not os.path.exists(args.source_folder+title+"_"+str(FRAME//fps_num)):
                os.makedirs(args.source_folder+title+"_"+str(FRAME//fps_num))
            while(video.isOpened()):
                ret, image = video.read()
                #current_time = time.time() - prev_time
                #print(count)
                if (ret is True) and (count % fps_num == 0) : # 지정한 프레임마다 뽑아낼 수 있도록
                #if(int(video.get(1)) % fps == 0): #앞서 불러온 fps 값을 사용하여 1초마다 추출
                    #prev_time = time.time()
                    cv2.imwrite(args.source_folder+title+"_"+str(FRAME//fps_num)+"/%s.png" % str(number), image)
                    number+=1
                count += 1
                if count >= total_frame-1:
                    break;
            base_dir = args.source_folder + title+"_"+str(FRAME//fps_num)
        print("전체 ",count,"장 중 ", number, "장을 출력했습니다.")
        print(title, "  사진으로 변환 완료했습니다.")
        video.release()



    # 이미지가 저장된 폴더 경로와 이동할 폴더 경로를 지정합니다.
    source = args.source_folder+args.name
    destination = args.destination_folder +  "/Interval_" + str(args.interval)
    os.makedirs(destination, exist_ok=True)
    # 이동할 이미지 파일 이름을 리스트로 지정합니다.
    img_lst = os.listdir(source)[int(args.start):int(args.end)] # 해당 영상에 특정 부분을 가져올 수 있도록 한다.
    
    # 저장 폴더 명 설정
    dest_lst = os.listdir(destination)
    if len(dest_lst)==0:
        dest_max=0
    else:
        dest_max = max([int(i) for i in dest_lst]) # 저장 파일에 있는 폴더 명 중 가장 큰 값을 가져온다.

    # 트리플렛 셋으로 이미지를 묶어준다.
    for i in range(len(img_lst[:-7])):
        if args.interval==2:
            lst = [os.path.join(source, o) for o in img_lst[i:i+5:2]] # 해당 이미지의 경로 설정
        elif args.interval==3:
            lst = [os.path.join(source, o) for o in img_lst[i:i+7:3]] # 해당 이미지의 경로 설정
        elif args.interval==1:
            lst = [os.path.join(source, o) for o in img_lst[i:i+3]] # 해당 이미지의 경로 설정
        #print(lst)
        # 저장 폴더 생성
        destination_name = destination + "/"+str(dest_max)
        if not os.path.exists(destination_name):
            os.makedirs(destination_name)
        for c, img in enumerate(lst):
            shutil.copy(img, destination_name+"/"+str(c+1)+".png" )
        # 폴더명 변경
        dest_max+=1

if __name__=="__main__":
    main()
