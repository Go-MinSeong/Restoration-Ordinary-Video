### test on SIDD ###
# python3 test/test_sidd.py --input_dir /home/kms990321/Uformer/datasets/denoising/DND/input5 --result_dir /home/kms990321/Uformer/results/denoising/SIDD2/ --weights /home/kms990321/Uformer/sid_Uformer_B.pth

### test on DND ###
# python test/test_dnd.py --input_dir /home/kms990321/Uformer/datasets/denoising/DND/input1 --result_dir ./results/denoising/DND1/ --weights /home/kms990321/Uformer/sid_Uformer_B.pth


### test on GoPro ###
python test/test_gopro_hide.py --input_dir /home/kms990321/DiffBIR/project/frames_inpaint --result_dir /home/kms990321/DiffBIR/project/frames_blur --weights /home/kms990321/Uformer/Uformer_B.pth

### test on HIDE ###
# python3 test/test_gopro_hide.py --input_dir ../datasets/deblurring/HIDE/test/ --result_dir ./results/deblurring/HIDE/Uformer_B/ --weights ./logs/motiondeblur/GoPro/Uformer_B/models/model_best.pth

### test on RealBlur ###
# python3 test/test_realblur.py --input_dir ../datasets/deblurring/ --result_dir ./results/deblurring/ --weights ./logs/motiondeblur/GoPro/Uformer_B/models/model_best.pth

