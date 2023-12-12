# Efficient Video Deblurring Guided by Motion Magnitude
Official repository of MMP-RNN (ECCV 2022) [Paper](https://arxiv.org/abs/2207.13374)  
Estimate pixel-wise blur level (MMP) first and video deblur.  
<div align="center"><img src="https://user-images.githubusercontent.com/11170161/178935637-6bb6a25a-dc67-4d5e-9c3c-f7f31be41085.png" width="480"></div>

## Requirements
Pytorch 1.8, Cuda 11.1  
lmdb, tqdm, thop, scipy, opencv, scikit-image, tensorboard

## Preparing ground truth motion magnitude prior
The ground truth MMP is generated from high-frequency sharp frames during exposure time. Optical flows between sharp frames are estimated by  [RAFT](https://github.com/princeton-vl/RAFT). 

<div align="center"><img src="https://user-images.githubusercontent.com/11170161/178949402-2de1df49-4fd8-481c-a4c2-8a0c534fa0fe.png" width="480"></div>

## Learning MMP
Using a UNet-like structure to learn MMP.

## MMP-RNN
Ultilizing MMP for video deblurring by merging into an RNN.

## Citation
```
@inproceedings{wang2022MMP,
  title={Efficient Video Deblurring Guided by Motion Magnitude},
  author={Wang, Yusheng and Lu, Yunfan and Gao, Ye and Wang, Lin and Zhong, Zhihang and Zheng, Yinqiang and Yamashita, Atsushi},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## Contact
Email of Yusheng Wang: wang@robot.t.u-tokyo.ac.jp

