CUDA_VISIBLE_DEVICES=1 \
python inference/colorization_pipline.py \
	--input /home/kms990321/DiffBIR/project/frames_color_before --output /home/kms990321/DiffBIR/project/frames_color_after \
	--model_path modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt