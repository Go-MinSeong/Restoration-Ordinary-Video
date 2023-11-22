# # import os
# # import random
# # from PIL import Image

# # # 원본 이미지 파일 경로
# # image_path = "/home/kms990321/Uformer/datasets/denoising/DND/input1/192.168.57.140_20231005223849.jpg"

# # # 저장할 디렉토리 경로
# # output_directory = "/home/kms990321/Uformer/datasets/denoising/DND/input2"

# # # 원본 이미지 열기
# # original_image = Image.open(image_path)

# # # 10장의 이미지를 랜덤하게 잘라서 저장
# # for i in range(20):
# #     # 이미지를 랜덤한 위치에서 512x512 크기로 잘라냅니다.
# #     width, height = original_image.size
# #     x = random.randint(0, width - 512)
# #     y = random.randint(0, height - 512)
# #     cropped_image = original_image.crop((x, y, x + 512, y + 512))

# #     # 잘라낸 이미지를 저장
# #     output_path = os.path.join(output_directory, f"cropped_image_{i}.jpg")
# #     cropped_image.save(output_path)
# #     print(f"이미지 저장: {output_path}")


import os
import torch
from scipy.io import savemat
import numpy as np
from PIL import Image

# 작업 디렉토리 및 .mat 파일 경로 설정
folder_path = '/home/kms990321/Uformer/datasets/denoising/DND/input5/input'  # 여기에 폴더 경로를 입력하세요
output_path = '/home/kms990321/Uformer/datasets/denoising/DND/input5/input.mat'  # .mat 파일을 저장할 경로 및 파일 이름을 입력하세요

# 폴더 내의 JPG 파일 목록 가져오기 및 정렬
jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
jpg_files.sort()

# 이미지를 텐서로 변환하고 1차원으로 쌓기
image_data = []
for jpg_file in jpg_files:
    img = Image.open(os.path.join(folder_path, jpg_file))
    img = np.array(img)
    
    # 이미지를 PyTorch 텐서로 변환하고 unsqueeze(0)을 사용하여 차원을 추가
    img_tensor = torch.from_numpy(img).unsqueeze(0)
    
    image_data.append(img_tensor)

# 텐서들을 1차원으로 쌓기
stacked_data = torch.cat(image_data, dim=0)
stacked_data = stacked_data.unsqueeze(0)
# .mat 파일로 저장
mat_data = {'ValidationNoisyBlocksSrgb': stacked_data.numpy()}  # PyTorch 텐서를 NumPy 배열로 변환
savemat(output_path, mat_data)

print(f'{len(jpg_files)}개의 JPG 파일이 {output_path}에 저장되었습니다.')


# from PIL import Image

# # 이미지 파일 경로
# input_image_path = '/home/kms990321/Uformer/datasets/deblurring/GoPro/test1/groundtruth/8448_sub_0.png'
# # 저장할 이미지 파일 경로
# output_image_path = '/home/kms990321/Uformer/datasets/deblurring/GoPro/test1/groundtruth/8448_sub_1.png'

# # 원본 이미지 열기
# image = Image.open(input_image_path)

# # 원본 이미지의 크기
# width, height = image.size

# width = width - 400
# height = height + 50
# # 오른쪽 가운데 부분의 좌표 계산
# left = width - 480
# top = (height - 360) / 2
# right = width
# bottom = (height + 360) / 2

# # 오른쪽 가운데 부분 자르기
# cropped_image = image.crop((left, top, right, bottom))

# # 자른 이미지 저장
# cropped_image.save(output_image_path)