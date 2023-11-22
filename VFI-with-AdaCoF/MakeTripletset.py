import shutil
import os
import argparse

parser = argparse.ArgumentParser(description='From Youtube to triplet data')

parser.add_argument('--name', type=str, default=None)
parser.add_argument('--start', type=str, default="0")
parser.add_argument('--end', type=str, default="1")
parser.add_argument('--interval', type=int, default=2) 
parser.add_argument('--destination_folder', type=str, default='/home/work/capstone/Final/YouTube')
parser.add_argument('--source_folder', type=str, default='/home/work/capstone/Final/interpolate_video/Before')

def main():
    args = parser.parse_args()

    # 이미지가 저장된 폴더 경로와 이동할 폴더 경로를 지정합니다.
    source = args.source_folder+"/"+args.name
    destination = args.destination_folder + "/Interval_" + str(args.interval)
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
