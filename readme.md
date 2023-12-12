<div align="center">

# <span style="color: #ff953e"> Restoration Antique Movie

**Project Duration [ 2023.11 ]**

</div>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
<div align="center">

## <span style="color: #ff953e"> **Objective**


**옛날 영상을 고품질의 영상으로 변환한다**

**Convert old footage into high-quality video**
  
  **부모님 결혼식 영상을 현대화하여 선물하기**
  
**Modernize and gift your parents' wedding video**

</div>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
<div align="center">


## <span style="color: #ff953e"> **Contribution**

</div>

- 사실 어머니의 생신을 맞이하여 한 프로젝트이기에, 부모님이 좋아하실 것으로 예상된다.
  
  I actually did this project for my mom's birthday, so I know she'll love it.
  
- 옛날 영상의 복원 프로세스를 하나의 프로세스로 통합함으로써 다른 저품질의 영상도 쉽게 고품질의 영상으로 복원 가능하다.

    By consolidating the process of restoring old footage into one process, you can easily restore other low-quality footage to high quality.

- Video Frame Interpolation의 경우, 직접 만든 Model로 이를 사용함으로써 성취감을 키운다.

    For Video Frame Interpolation, you get a sense of accomplishment by using it with your own model.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
<div align="center">

## <span style="color: #ff953e"> **Process**

</div>
각 기술들의 간단한 원리와 출처를 적을 예정입니다.

I'll write down the simple principles and sources for each technique.

총 5가지의 기술을 이용하여 이미지를 제작할 예정이며 사용할 기술들은 다음과 같다.

We'll be using a total of five different techniques to create the image, and they are as follows

**1.** **Image Inpainting**을 통하여 기존 비디오에 있던 불완전한 영역을 자연스러운 영역으로 교체한다. 예를 들어, 시간 표시로 가려져 있는 영역 혹은 외곽 끝 부분의 부자연스러운 영역들을 제거하고 자연스러운 부분으로 교체 진행
  
**2.** **Image Deblurring**을 이용하여 옛날 카메라 기술의 부족으로 생긴 Blur한 Frame을 자연스럽게 처리해준다.

**3.** **Image Colorization**을 이용하여 흑백으로 표현되어 있던 Frame을 색감 처리를 진행한다.

**4.** **Super Resolution**을 이용하여 기존 화질 360X240의 영상에서 1280X720의 영상으로 교체한다.

**5.** **Video Frame Interpolation**을 이용하여 기존 FPS 25 => FPS 50의 영상으로 제작한다.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
<div align="center">

### <span style="color: #ff953e"> **Tech**

#### [Image Inpainting](https://github.com/fenglinglwb/MAT)
##### MAT: Mask-Aware Transformer for Large Hole Image Inpainting

#### [Image Deblurring](https://github.com/ZhendongWang6/Uformer)
##### Uformer: A General U-Shaped Transformer for Image Restoration

#### [Image Colorization](https://github.com/piddnad/DDColor)
##### DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders

#### [Super Resolution](https://github.com/XPixelGroup/DiffBIR)
##### DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior

#### [Video Frame Interpolation](https://github.com/Go-MinSeong/VFI-with-AdaCoF)
##### SF-AdaCoF: Siamese Frane AdaCoF for Kernel based Video Frame Interpolation
</div>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<div align="center">

## <span style="color: #ff953e"> **Result**

</div>

In this part, I'll show you the result of each process.
The result image consists of a before(Left) and after(Right) image.

### <span style="color: #fff5b1"> **Image Inpainting**

![stronghold logo](frames_versus_instance/inpainting.png)

- Left image is original images the other one is inpainted image. I use inpainting technique for fill in the ends of the four sides. For instance, you can see blur region and black region in left image. But there is not a awkward region anywhere in right one. 
- I set a mask image that has 20pixel mask section every side. The inpainting model predicted the section naturally. 

### <span style="color: #fff5b1"> **Image Deblurring**

![stronghold logo](frames_versus_instance/deblurring.png)

- Left image is original images the other one is deblurred image. I guess that My parents wedding video was shot by one person walking around. So for most images, the blurring exists. To solve this problem,  I utilize deblurring technique.
- For example, I've highlighted the flowerpot in the image. You can see clearer image on the right side.
- Before applying this method, In fact, I want to deblurr for human. But it doesn't not work well. I'm looking forward to solving this problem.

### <span style="color: #fff5b1"> **Image Colorization**

![stronghold logo](frames_versus_instance/colorization.png)

- All images in video are not in black and white. But there are some like that. I think that black and white conversion techniques are used to take a video.
- But I don't like it. Then I decided to colorize images. 
- I think that colorized image is more better than before.

### <span style="color: #fff5b1"> **Image Super Resolution**

![stronghold logo](frames_versus_instance/super_resolution.png)

- You know, there are not only image restoration method, but also video restoration method. As I know video restoration methods utilize subsequent images, But thare are a lot of edit point that changs viewpoint so I conclude that image restoration method is more suitable for this work.
- Original video's resolution is 720X480, but when I just resize it to 360X240 there is a any comression losing information. So, I suppose the video's real resolution is 360X240. All process is worked on this assumption. 
- Now, 360X240 resolution is too low to watch theese days. Therefore, I apply to Super resolution in this process. I increased the resolution by 4X.
- We can see a higher quality image. However, we can still see some pixelations in some area.
- After super resolution, Thing to keep in mind is we need bigger VRAM, stronger GPU and so on. Because According to increase resolution, there are more calculation. This is the reason that I place this method in almost last section.

### <span style="color: #fff5b1"> **Video Frame Interpolation**

<p float="center">
  <img src="result/result.gif" alt="part of video" height="240px" width="360px" />

</p>

- Finally, We arrive last method, Video Frame Interpolation. The original video has 30FPS(frame per second). I know, 30FPS is not bad but I don't want moderate quality. Hence, I make a 60FPS video. Of course, I could make 90FPS, 120FPS but it is need a lot of memory.
- Expecially, This part is crucial for me. Because used VFI model is made by me. And so I cound't choose any other model in task. Additionally, most of all this project is planned to present for my parents. Then, I think that result by my own model is special for both me and my parents.
- I show you part of the result through gif format above the paragraph.



![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


<div align="center">

## <span style="color: #ff953e"> **What needs improvement**

</div>

1.  현재, 다양한 Task 절차를 거치는데 이 Task간 순서에 따른 결과 실험이 필요할 것으로 보인다.  

    Currently, we're going through a variety of task procedures and will need to experiment with the results of the ordering of these tasks.  
    <br/>
2.  테스크 간 모델의 인퍼런스를 위해 이미지의 사이즈가 훼손되어지는 경우가 발생했고, 이는 정보 손실이 있었을 것으로 생각된다.  

    In some cases, the size of the image was compromised to allow for cross-task model references, which may have resulted in information loss.<br/>
    <br/>
3.  Colorization Task에서 각 이미지 마다의 Color Embedding을 진행하였다. 이는, 연속된 frame간 부자연스러운 색깔 변환을 일으킨다. 이를 해결하기 위해 Video Colorization Model을 사용하였으나, 사용한 single image colorization보다 이미지 마다의 정확도가 떨어져 지금은 Single Image Colorization model을 사용하며 이는 추후 개선 작업이 필요하다.

    In the Colorization Task, the color is embedded in each image. This causes unnatural color conversion between consecutive frames. To solve this problem, we used Video Colorization Model, but the accuracy of each image is lower than single image colorization, so we use Single Image Colorization model, which needs to be improved in the future.
<br/><br/>
4.  Super Resolution Task에서 각 이미지의 Inference 시간이 오래 걸려 해당 Model의 2stage(Stable Diffusion) 부분을 제외하고 진행하였다. 이는 추후 속도 개선 후, 추가 적용이 필요하다.

    In the Super Resolution Task, the inference time for each image was long, so we excluded the second stage (Stable Diffusion) part of the model. This will be applied in the future after improving the speed.
<br/><br/>
5.  전제적인 프로세스 실행 방식을 정리해야 한다.
    I need to clean all process to execute.
