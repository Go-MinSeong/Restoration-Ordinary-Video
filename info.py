from moviepy.editor import VideoFileClip
from PIL import Image
import os
from tqdm import tqdm
import shutil
import random

def get_video_info(file_path):
    try:
        video_clip = VideoFileClip(file_path)
        print("Duration:", video_clip.duration)
        print("FPS:", video_clip.fps)
        print("Size:", video_clip.size)
        print("Video Codec:", video_clip.fps)

        # 오디오 정보에 접근
        audio = video_clip.audio
        if audio:
            print("Audio Sample Rate:", audio.fps)
            print("Audio Channels:", audio.nchannels)
        else:
            print("No audio in the video.")

    except Exception as e:
        print(f"Error: {e}")

get_video_info(file_path = "movie.mp4")

def resize_video(input_path, output_path, target_resolution):
    try:
        video_clip = VideoFileClip(input_path)
        resized_clip = video_clip.resize(height=target_resolution[1], width=target_resolution[0])
        resized_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True)
    except Exception as e:
        print(f"Error: {e}")

# resize_video(input_path = "movie.mp4", output_path = "output_video.mp4", target_resolution = (360, 240))


def change_fps(input_path, output_path, target_fps):
    try:
        video_clip = VideoFileClip(input_path)
        modified_clip = video_clip.set_fps(target_fps)
        modified_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True)
    except Exception as e:
        print(f"Error: {e}")

# change_fps(input_path = "output_video.mp4" , output_path = "output_video_(fps25).mp4", target_fps = 25)

def extract_frames(input_path, output_folder):
    try:
        video_clip = VideoFileClip(input_path)
        frames_folder = os.path.join(output_folder)
        
        # 프레임을 저장할 폴더가 없다면 생성
        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder)

        # 비디오의 모든 프레임을 이미지로 저장
        for i, frame in enumerate(video_clip.iter_frames(fps=video_clip.fps)):
            frame_path = os.path.join(frames_folder, f'{i:04d}.png')
            frame_img = Image.fromarray(frame)
            frame_img.save(frame_path)

        print(f"All frames are saved in: {frames_folder}")

    except Exception as e:
        print(f"Error: {e}")

# extract_frames(input_path = "output_video_(fps25).mp4", output_folder  = "frames")


def resize_image(input_path, output_path, target_size=(512, 512)):
    try:
        # 이미지 열기
        with Image.open(input_path) as img:
            # 이미지 리사이즈
            resized_img = img.resize(target_size)

            # 리사이즈된 이미지 저장
            resized_img.save(output_path)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def resize_images_in_folder(input_folder, output_folder, target_size=(512, 512)):
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 입력 폴더에서 파일 목록 가져오기
    file_list = os.listdir(input_folder)

    for filename in tqdm(file_list):
        # 파일 경로 및 이름 구성
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 이미지 리사이즈 함수 호출
        resize_image(input_path, output_path, target_size)

# resize_images_in_folder("/home/kms990321/DiffBIR/project/frames", "/home/kms990321/DiffBIR/project/frames512")


def black_image_move(input_path, output_path, output_path2, target_num =[1882,3722]):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path2, exist_ok=True)
    img_lst = os.listdir(input_path)
    target_lst = ['{:04}'.format(target_num[0]+x) for x in range(target_num[1] - target_num[0])]
    img_lst1 = [x for x in img_lst if x.split("_")[1].split(".")[0] in target_lst]
    img_lst2 = [x for x in img_lst if x.split("_")[1].split(".")[0] not in target_lst]
    for img_path in tqdm(img_lst1):
        shutil.copy(os.path.join(input_path, img_path), os.path.join(output_path, img_path))
    for img_path in tqdm(img_lst2):
        shutil.copy(os.path.join(input_path, img_path), os.path.join(output_path2, img_path))

# black_image_move("/home/kms990321/DiffBIR/project/frames_blur", "/home/kms990321/DiffBIR/project/frames_color_before", "/home/kms990321/DiffBIR/project/frames_color_after")


def resize_images(input_folder, output_folder, target_size=(360, 240)):
    # 입력 폴더 내의 모든 파일 가져오기
    file_list = os.listdir(input_folder)
    os.makedirs(output_folder, exist_ok=True)
    for file_name in tqdm(file_list):
        # 이미지 파일인지 확인
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 이미지 경로 설정
            input_path = os.path.join(input_folder, file_name)
            
            # 이미지 열기
            image = Image.open(input_path)
            
            # 이미지 크기 조정
            resized_image = image.resize(target_size)
            
            # 출력 이미지 경로 설정
            output_path = os.path.join(output_folder, file_name)
            
            # 크기 조정된 이미지 저장
            resized_image.save(output_path)

    print("이미지 크기 조정이 완료되었습니다.")

# 사용 예시
input_folder = "/home/kms990321/DiffBIR/project/frames_color_after"
output_folder = "/home/kms990321/DiffBIR/project/frames_SR_before"
target_size = (360, 240)

# resize_images(input_folder, output_folder, target_size)


def merge_random_images(folder1, folder2, output_folder, num_images=30):
    os.makedirs(output_folder, exist_ok=True)
    # 폴더에서 파일 목록 가져오기
    files1 = [f.split(".")[0] for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))]
    files2 = [f.split(".")[0] for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))]

    # 두 폴더에서 공통으로 존재하는 파일 찾기
    common_files = list(set(files1) & set(files2))

    # 랜덤하게 파일 선택 (최대 10개)
    selected_files = random.sample(common_files, min(num_images, len(common_files)))

    # 이미지 합치기 및 저장
    for file in selected_files:
        path1 = os.path.join(folder1, file)
        path2 = os.path.join(folder2, file)

        try:
            image1 = Image.open(path1+".png")
        except:
            image1 = Image.open(path1+".PNG")
        try:
            image2 = Image.open(path2+".png")
        except:
            image2 = Image.open(path2+".PNG")

        width1, height1 = image1.size
        width2, height2 = image2.size
        if width1 != width2 or height1 != height2:
            image1 = image1.resize((width2, height2))

        # 이미지를 가로로 합치기
        merged_image = Image.new('RGB', (image1.width + image2.width, max(image1.height, image2.height)))
        merged_image.paste(image1, (0, 0))
        merged_image.paste(image2, (image1.width, 0))

        # 저장 경로 설정 및 저장
        v1 = folder1.split("/")[-1]; v2 = folder2.split("/")[-1]
        output_path = os.path.join(output_folder, f"{v1}_{v2}_{file}.png")
        merged_image.save(output_path)

folder1 = "/home/kms990321/DiffBIR/project/frames_SR"
folder2 = "/home/kms990321/DiffBIR/project/frames_SR2"
output_folder = "/home/kms990321/DiffBIR/project/frames_versus"
# merge_random_images(folder1, folder2, output_folder)


def copy_and_rename_png_files(source_folder, destination_folder, suffix='_new'):
    # 대상 폴더가 존재하지 않으면 생성
    os.makedirs(destination_folder, exist_ok=True)

    # source 폴더에 있는 모든 PNG 파일 찾기
    for root, _, files in os.walk(source_folder):
        for file in tqdm(files):
            if file.lower().endswith('.png'):
                source_path = os.path.join(root, file)

                # 파일 이름 수정 (여기서는 파일 확장자 앞에 suffix를 추가하는 예제)
                num = int(os.path.splitext(file)[0][1:]) + 2263
                new_name = f"{num:04d}.png"
                
                # 파일 복사
                destination_path = os.path.join(destination_folder, new_name)
                shutil.copy2(source_path, destination_path)

source_folder = "/home/kms990321/DiffBIR/project/TCVC/stage2_test_results/1"
destination_folder = "/home/kms990321/DiffBIR/project/data/frames_color_after"

# copy_and_rename_png_files(source_folder, destination_folder)


from pathlib import Path

def distribute_images_to_folders_by_count(source_folder, destination_parent_folder, images_per_folder):
    # source_folder: 이미지가 있는 폴더
    # destination_parent_folder: 하위 폴더들을 포함하는 상위 폴더
    # images_per_folder: 각 폴더당 이동시킬 이미지 개수

    # source_folder 내의 이미지 파일 목록 얻기
    image_files = [f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    image_files = sorted(image_files, key=lambda x: int(x.split(".")[0]))

    # 이미지 파일 개수 세기
    num_images = len(image_files)

    # 폴더 개수 계산
    num_folders = (num_images + images_per_folder - 1) // images_per_folder

    # destination_parent_folder가 없다면 생성
    if not os.path.exists(destination_parent_folder):
        os.makedirs(destination_parent_folder)

    # 이미지를 각 폴더에 차례대로 이동
    for i in range(num_folders):
        subfolder_name = f"subfolder_{i+1}"
        subfolder_path = os.path.join(destination_parent_folder, subfolder_name)

        # 하위 폴더가 없다면 생성
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        start_index = i * images_per_folder
        end_index = min((i + 1) * images_per_folder, num_images)

        for j in range(start_index, end_index):
            image = image_files[j]
            source_path = os.path.join(source_folder, image)
            destination_path = os.path.join(subfolder_path, image)
            shutil.move(source_path, destination_path)
            print(f"{image} 이동 완료 -> {subfolder_path}")

# 사용 예제
source_folder = '/home/kms990321/DiffBIR/project/data/frame_deblur240'  # 실제 폴더 경로로 변경
destination_parent_folder = '/home/kms990321/DiffBIR/project/data/frame_deblur240'  # 실제 폴더 경로로 변경
images_per_folder = 128  # 각 폴더당 이동시킬 이미지 개수

# distribute_images_to_folders_by_count(source_folder, destination_parent_folder, images_per_folder)

def move_images_from_subfolders(source_parent_folder, destination_folder):
    # source_parent_folder: 이미지가 있는 하위 폴더들을 포함하는 상위 폴더
    # destination_folder: 이미지를 이동시킬 폴더

    # source_parent_folder 내의 모든 하위 폴더 얻기
    subfolders = [f.path for f in os.scandir(source_parent_folder) if f.is_dir()]

    # destination_folder가 없다면 생성
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 각 하위 폴더의 이미지를 상위 폴더로 이동
    for subfolder in subfolders:
        image_files = [f for f in os.listdir(subfolder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        for image in image_files:
            source_path = os.path.join(subfolder, image)
            destination_path = os.path.join(destination_folder, image)
            shutil.move(source_path, destination_path)
            print(f"{image} 이동 완료 -> {destination_folder}")

# 사용 예제
source_parent_folder = '/home/kms990321/DiffBIR/project/data/frame_deblur240'  # 실제 상위 폴더 경로로 변경
destination_folder = '/home/kms990321/DiffBIR/project/data/frame_deblur240'  # 실제 폴더 경로로 변경

# move_images_from_subfolders(source_parent_folder, destination_folder)