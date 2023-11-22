import torch
import cupy_module.adacof as adacof
import sys
from torch.nn import functional as F
from utility import CharbonnierFunc, moduleNormalize # CharbonnierFunction과 Normalize 함수를 불러옴


def make_model(args):
    return AdaCoFNet(args).cuda()

# adacof는 kernel을 학습하는 것이 아닌 output으로 가중치를 얻어야 함.
class KernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(KernelEstimation, self).__init__()
        self.kernel_size = kernel_size # 마지막 output channel들은 몇개로 할지 지정.

        # Basic한 block은 Conv -> ReLu 3번 거치고, kernel size는 3 , padding은 1
        def Basic(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        # Upsample block은 Upsample -> Conv -> ReLU 순서
        def Upsample(channel):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 작은 크기의 Feature를 크게 변경시킬 때 사용.
                torch.nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        
        # offset에서 ks는 최종 ouput channel 수
        def Subnet_offset(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1)
            )

        # weight에서 ks는 최종 output channel 수
        def Subnet_weight(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.Softmax(dim=1)
            )

        def Subnet_occlusion():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
                torch.nn.Sigmoid()
            )

        # Flow estimation은 3x3 conv -> 1x1 conv로 진행. 좀 더 공간정보를 많이 뽑을 수 있도록 함.
        def Flow_estimation(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=3, stride=1, padding = 1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, stride=1),
                torch.nn.ReLU(inplace=False),
            )

        # conv에서는 convolution + average pooling, Deconv에서는 conv + upsample
        # 이전과 이후 시점의 이미지가 병렬로 들어가는 구역에서 채널을 절반으로 줄인거 - 1 해서 flow feature map이 들어갈 수 있게 해줌.
        self.moduleConv1 = Basic(3, 14) # 6,32 -> 3 ,16 -> 3, 15
        self.moduleConv1_1 = Basic(3, 14)
        self.modulePool1 = torch.nn.Conv2d(in_channels=16,out_channels=16, kernel_size=2, stride=2)
        self.modulePool1_1 = torch.nn.Conv2d(in_channels=16,out_channels=16, kernel_size=2, stride=2)
        self.flowconv1_FD = Flow_estimation(14,1)
        self.flowconv1_pr = Flow_estimation(14,1)
        self.flowconv1_ne = Flow_estimation(14,1)

        self.moduleConv2 = Basic(16, 30) # 32,64 -> 16 ,32 -> 16,31
        self.moduleConv2_1 = Basic(16, 30)
        self.modulePool2 = torch.nn.Conv2d(in_channels=32,out_channels=32, kernel_size=2, stride=2)
        self.modulePool2_1 = torch.nn.Conv2d(in_channels=32,out_channels=32, kernel_size=2, stride=2)
        self.flowconv2_FD = Flow_estimation(30,1)
        self.flowconv2_pr = Flow_estimation(30,1)
        self.flowconv2_ne = Flow_estimation(30,1)

        self.moduleConv3 = Basic(32,62) # 64,128 -> 32,64 -> 32,63
        self.moduleConv3_1 = Basic(32,62)
        self.modulePool3 = torch.nn.Conv2d(in_channels=64,out_channels=64, kernel_size=2, stride=2)
        self.modulePool3_1 = torch.nn.Conv2d(in_channels=64,out_channels=64, kernel_size=2, stride=2)
        self.flowconv3_FD = Flow_estimation(62,1)
        self.flowconv3_pr = Flow_estimation(62,1)
        self.flowconv3_ne = Flow_estimation(62,1)

        self.moduleConv4 = Basic(64, 126) # 128,256 -> 64,128 -> 64,127
        self.moduleConv4_1 = Basic(64, 126) # 128,256 -> 64,128 -> 64,127
        self.modulePool4 = torch.nn.Conv2d(in_channels=128,out_channels=128, kernel_size=2, stride=2)
        self.modulePool4_1 = torch.nn.Conv2d(in_channels=128,out_channels=128, kernel_size=2, stride=2)
        self.flowconv4_FD = Flow_estimation(126,1)
        self.flowconv4_pr = Flow_estimation(126,1)
        self.flowconv4_ne = Flow_estimation(126,1)

        self.moduleConv5 = Basic(128,254) # 256,512 -> 128,256 -> 128, 255
        self.moduleConv5_1 = Basic(128,254) # 256,512 -> 128,256 -> 128, 255
        self.modulePool5 = torch.nn.Conv2d(in_channels=256,out_channels=256, kernel_size=2, stride=2)
        self.modulePool5_1 = torch.nn.Conv2d(in_channels=256,out_channels=256, kernel_size=2, stride=2)
        self.flowconv5_FD = Flow_estimation(254,1)
        self.flowconv5_pr = Flow_estimation(254,1)
        self.flowconv5_ne = Flow_estimation(254,1)

        # Decoder 단에서는 그냥 채널 수 그대로 진행. -> 어차피 concat 되면 똑같음.

        self.moduleDeconv5 = Basic(256,256) # 512,512 -> 256,256
        self.moduleDeconv5_1 = Basic(256,256)
        self.moduleUpsample5 = Upsample(256) # 512 -> 256
        self.moduleUpsample5_1 = Upsample(256) # 512 -> 256

        self.moduleDeconv4 = Basic(256,128) # 512,256 -> 256,128
        self.moduleDeconv4_1 = Basic(256,128)
        self.moduleUpsample4 = Upsample(128) # 256 -> 128
        self.moduleUpsample4_1 = Upsample(128) # 512 -> 256

        self.moduleDeconv3 = Basic(128,64) # 256,128 -> 128,64
        self.moduleDeconv3_1 = Basic(128,64)
        self.moduleUpsample3 = Upsample(64) # 128 -> 64
        self.moduleUpsample3_1 = Upsample(64) # 128 -> 64

        self.moduleDeconv2 = Basic(64,32) # 128,64 -> 64,32
        self.moduleDeconv2_1 = Basic(64,32)
        self.moduleUpsample2 = Upsample(32) # 64 -> 32
        self.moduleUpsample2_1 = Upsample(32) # 64 -> 32
        
        ## kernel_size에 제곱을 해주어 25로 진행하는듯. 최종적인인 output channel의 개수는 25개인듯.
        self.moduleWeight1 = Subnet_weight(self.kernel_size ** 2) 
        self.moduleAlpha1 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta1 = Subnet_offset(self.kernel_size ** 2)
        self.moduleWeight2 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha2 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta2 = Subnet_offset(self.kernel_size ** 2) 
        self.moduleOcclusion = Subnet_occlusion() # 이게 V

    # 여기서 이미지가 하나씩 들어가도록 해주어야함.
    def forward(self, rfield0, rfield2):
        
        tensorPrev = rfield0
        tensorNext = rfield2

       
        tensorConv1_pr = self.moduleConv1(tensorPrev)
        tensorConv1_ne = self.moduleConv1_1(tensorNext)

        # frame differencing feature map 생성
        tensor_FD = torch.abs(tensorConv1_ne - tensorConv1_pr)
        tensorFlowConv1_FD = self.flowconv1_FD(tensor_FD)

        # 원래 Feature map flow conv block 통과 이후 Frame Differencing과 element-wise product
        tensorFlowConv1_pr = self.flowconv1_pr(tensorConv1_pr)
        tensorFlowConv1_ne = self.flowconv1_ne(tensorConv1_ne)
        tensorFlowConv1_pr = tensorFlowConv1_pr * tensorFlowConv1_FD
        tensorFlowConv1_ne = tensorFlowConv1_ne * tensorFlowConv1_FD

        # flow와 각 시점에서의 feature map들을 concat -> conv와 pooling 된 것들을 따로 합쳐주어야함 -> Decoder 단에서 Deconvolution하기 때문.
        tensorConv1_pr = torch.cat([tensorConv1_pr,tensorFlowConv1_pr,tensorFlowConv1_ne],1)
        tensorConv1_ne = torch.cat([tensorConv1_ne,tensorFlowConv1_pr,tensorFlowConv1_ne],1)

        # flow와 conv를 먼저 합치고 진행함.
        tensorPool1_pr = self.modulePool1(tensorConv1_pr)
        tensorPool1_ne = self.modulePool1_1(tensorConv1_ne)


        tensorConv2_pr = self.moduleConv2(tensorPool1_pr)
        tensorConv2_ne = self.moduleConv2_1(tensorPool1_ne)
        

        # flow feature map 생성
        tensor_FD = torch.abs(tensorConv2_ne - tensorConv2_pr)
        tensorFlowConv2_FD = self.flowconv2_FD(tensor_FD)

        # 원래 Feature map flow conv block 통과 이후 Frame Differencing과 element-wise product
        tensorFlowConv2_pr = self.flowconv2_pr(tensorConv2_pr)
        tensorFlowConv2_ne = self.flowconv2_ne(tensorConv2_ne)
        tensorFlowConv2_pr = tensorFlowConv2_pr * tensorFlowConv2_FD
        tensorFlowConv2_ne = tensorFlowConv2_ne * tensorFlowConv2_FD

        # flow와 각 시점에서의 feature map들을 concat
        tensorConv2_pr = torch.cat([tensorConv2_pr,tensorFlowConv2_pr,tensorFlowConv2_ne],1)
        tensorConv2_ne = torch.cat([tensorConv2_ne,tensorFlowConv2_pr,tensorFlowConv2_ne],1)

        tensorPool2_pr = self.modulePool2(tensorConv2_pr)
        tensorPool2_ne = self.modulePool2_1(tensorConv2_ne)

        tensorConv3_pr = self.moduleConv3(tensorPool2_pr)
        tensorConv3_ne = self.moduleConv3_1(tensorPool2_ne)

        # flow feature map 생성
        tensor_FD = torch.abs(tensorConv3_ne - tensorConv3_pr)
        tensorFlowConv3_FD = self.flowconv3_FD(tensor_FD)

        # 원래 Feature map flow conv block 통과 이후 Frame Differencing과 element-wise product
        tensorFlowConv3_pr = self.flowconv3_pr(tensorConv3_pr)
        tensorFlowConv3_ne = self.flowconv3_ne(tensorConv3_ne)
        tensorFlowConv3_pr = tensorFlowConv3_pr * tensorFlowConv3_FD
        tensorFlowConv3_ne = tensorFlowConv3_ne * tensorFlowConv3_FD

        # flow와 각 시점에서의 feature map들을 concat
        tensorConv3_pr = torch.cat([tensorConv3_pr,tensorFlowConv3_pr,tensorFlowConv3_ne],1)
        tensorConv3_ne = torch.cat([tensorConv3_ne,tensorFlowConv3_pr,tensorFlowConv3_ne],1)

        tensorPool3_pr = self.modulePool3(tensorConv3_pr)
        tensorPool3_ne = self.modulePool3_1(tensorConv3_ne)

        tensorConv4_pr = self.moduleConv4(tensorPool3_pr)
        tensorConv4_ne = self.moduleConv4_1(tensorPool3_ne)

        # flow feature map 생성
        tensor_FD = torch.abs(tensorConv4_ne - tensorConv4_pr)
        tensorFlowConv4_FD = self.flowconv4_FD(tensor_FD)

        # 원래 Feature map flow conv block 통과 이후 Frame Differencing과 element-wise product
        tensorFlowConv4_pr = self.flowconv4_pr(tensorConv4_pr)
        tensorFlowConv4_ne = self.flowconv4_ne(tensorConv4_ne)
        tensorFlowConv4_pr = tensorFlowConv4_pr * tensorFlowConv4_FD
        tensorFlowConv4_ne = tensorFlowConv4_ne * tensorFlowConv4_FD

        # flow와 각 시점에서의 feature map들을 concat
        tensorConv4_pr = torch.cat([tensorConv4_pr,tensorFlowConv4_pr,tensorFlowConv4_ne],1)
        tensorConv4_ne = torch.cat([tensorConv4_ne,tensorFlowConv4_pr,tensorFlowConv4_ne],1)

        tensorPool4_pr = self.modulePool4(tensorConv4_pr)
        tensorPool4_ne = self.modulePool4_1(tensorConv4_ne)

        tensorConv5_pr = self.moduleConv5(tensorPool4_pr)
        tensorConv5_ne = self.moduleConv5_1(tensorPool4_ne)

        # flow feature map 생성
        tensor_FD = torch.abs(tensorConv5_ne - tensorConv5_pr)
        tensorFlowConv5_FD = self.flowconv5_FD(tensor_FD)

        # 원래 Feature map flow conv block 통과 이후 Frame Differencing과 element-wise product
        tensorFlowConv5_pr = self.flowconv5_pr(tensorConv5_pr)
        tensorFlowConv5_ne = self.flowconv5_ne(tensorConv5_ne)
        tensorFlowConv5_pr = tensorFlowConv5_pr * tensorFlowConv5_FD
        tensorFlowConv5_ne = tensorFlowConv5_ne * tensorFlowConv5_FD

        # flow와 각 시점에서의 feature map들을 concat
        tensorConv5_pr = torch.cat([tensorConv5_pr,tensorFlowConv5_pr,tensorFlowConv5_ne],1)
        tensorConv5_ne = torch.cat([tensorConv5_ne,tensorFlowConv5_pr,tensorFlowConv5_ne],1)

        tensorPool5_pr = self.modulePool5(tensorConv5_pr)
        tensorPool5_ne = self.modulePool5_1(tensorConv5_ne)
        
        # 각각의 Deconv에서는 해당하는 Layer의 Conv tensor 값과 element-wise add해줌.
        tensorDeconv5_pr = self.moduleDeconv5(tensorPool5_pr)
        tensorUpsample5_pr = self.moduleUpsample5(tensorDeconv5_pr)

        tensorDeconv5_ne = self.moduleDeconv5_1(tensorPool5_ne)
        tensorUpsample5_ne = self.moduleUpsample5_1(tensorDeconv5_ne)

        tensorCombine_pr = tensorUpsample5_pr + tensorConv5_pr
        tensorCombine_ne = tensorUpsample5_ne + tensorConv5_ne

        tensorDeconv4_pr = self.moduleDeconv4(tensorCombine_pr)
        tensorUpsample4_pr = self.moduleUpsample4(tensorDeconv4_pr)

        tensorDeconv4_ne = self.moduleDeconv4_1(tensorCombine_ne)
        tensorUpsample4_ne = self.moduleUpsample4_1(tensorDeconv4_ne)

        tensorCombine_pr = tensorUpsample4_pr + tensorConv4_pr
        tensorCombine_ne = tensorUpsample4_ne + tensorConv4_ne

        tensorDeconv3_pr = self.moduleDeconv3(tensorCombine_pr)
        tensorUpsample3_pr = self.moduleUpsample3(tensorDeconv3_pr)

        tensorDeconv3_ne = self.moduleDeconv3_1(tensorCombine_ne)
        tensorUpsample3_ne = self.moduleUpsample3_1(tensorDeconv3_ne)

        tensorCombine_pr = tensorUpsample3_pr + tensorConv3_pr
        tensorCombine_ne = tensorUpsample3_ne + tensorConv3_ne

        tensorDeconv2_pr = self.moduleDeconv2(tensorCombine_pr)
        tensorUpsample2_pr = self.moduleUpsample2(tensorDeconv2_pr)

        tensorDeconv2_ne = self.moduleDeconv2_1(tensorCombine_ne)
        tensorUpsample2_ne = self.moduleUpsample2_1(tensorDeconv2_ne) 

        tensorCombine_pr = tensorUpsample2_pr + tensorConv2_pr
        tensorCombine_ne = tensorUpsample2_ne + tensorConv2_ne

        # print(f'중간 shape 확인 !! -> {tensorCombine_pr.shape}|||{tensorCombine_ne.shape}')

        tensorCombine = torch.cat([tensorCombine_pr,tensorCombine_ne],1)

        Weight1 = self.moduleWeight1(tensorCombine)
        Alpha1 = self.moduleAlpha1(tensorCombine)
        Beta1 = self.moduleBeta1(tensorCombine)
        Weight2 = self.moduleWeight2(tensorCombine)
        Alpha2 = self.moduleAlpha2(tensorCombine)
        Beta2 = self.moduleBeta2(tensorCombine)
        Occlusion = self.moduleOcclusion(tensorCombine)

        return Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, Occlusion


class AdaCoFNet(torch.nn.Module):
    def __init__(self, args): # args를 인자로 받아감
        super(AdaCoFNet, self).__init__()
        self.args = args
        self.kernel_size = args.kernel_size # args에서 kernel_size를 가져가고
        self.kernel_pad = int(((args.kernel_size - 1) * args.dilation) / 2.0) # padding은 kernel_size -1 * dilation(1) / 2로 진행.
        self.dilation = args.dilation

        self.get_kernel = KernelEstimation(self.kernel_size)

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad]) # 가장 가까운 픽셀값을 padding으로 진행

        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

    def forward(self, frame0, frame2): # B,C,H,W -> 세로,가로
        h0 = int(list(frame0.size())[2])
        w0 = int(list(frame0.size())[3])
        h2 = int(list(frame2.size())[2])
        w2 = int(list(frame2.size())[3])
        # 사이즈가 동일하지 않으면 끝내버림
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        h_padded = False
        w_padded = False
        if h0 % 32 != 0:
            pad_h = 32 - (h0 % 32)
            frame0 = F.pad(frame0, (0, 0, 0, pad_h), mode='reflect')
            frame2 = F.pad(frame2, (0, 0, 0, pad_h), mode='reflect')
            h_padded = True

        if w0 % 32 != 0:
            pad_w = 32 - (w0 % 32)
            frame0 = F.pad(frame0, (0, pad_w, 0, 0), mode='reflect')
            frame2 = F.pad(frame2, (0, pad_w, 0, 0), mode='reflect')
            w_padded = True
        Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, Occlusion = self.get_kernel(moduleNormalize(frame0), moduleNormalize(frame2))

        tensorAdaCoF1 = self.moduleAdaCoF(self.modulePad(frame0), Weight1, Alpha1, Beta1, self.dilation) # output이 들어있음.
        tensorAdaCoF2 = self.moduleAdaCoF(self.modulePad(frame2), Weight2, Alpha2, Beta2, self.dilation) # output이 들어있음.

        frame1 = Occlusion * tensorAdaCoF1 + (1 - Occlusion) * tensorAdaCoF2
        # 만약 패딩을 했다면 
        if h_padded:
            frame1 = frame1[:, :, 0:h0, :]
        if w_padded:
            frame1 = frame1[:, :, :, 0:w0]

        if self.training:
            # Smoothness Terms
            # 채널별로 총합이 1인 softmax가 적용된 weight와 offset vector를 적용.
            m_Alpha1 = torch.mean(Weight1 * Alpha1, dim=1, keepdim=True)
            m_Alpha2 = torch.mean(Weight2 * Alpha2, dim=1, keepdim=True)
            m_Beta1 = torch.mean(Weight1 * Beta1, dim=1, keepdim=True)
            m_Beta2 = torch.mean(Weight2 * Beta2, dim=1, keepdim=True)

            g_Alpha1 = CharbonnierFunc(m_Alpha1[:, :, :, :-1] - m_Alpha1[:, :, :, 1:]) + CharbonnierFunc(m_Alpha1[:, :, :-1, :] - m_Alpha1[:, :, 1:, :])
            g_Beta1 = CharbonnierFunc(m_Beta1[:, :, :, :-1] - m_Beta1[:, :, :, 1:]) + CharbonnierFunc(m_Beta1[:, :, :-1, :] - m_Beta1[:, :, 1:, :])
            g_Alpha2 = CharbonnierFunc(m_Alpha2[:, :, :, :-1] - m_Alpha2[:, :, :, 1:]) + CharbonnierFunc(m_Alpha2[:, :, :-1, :] - m_Alpha2[:, :, 1:, :])
            g_Beta2 = CharbonnierFunc(m_Beta2[:, :, :, :-1] - m_Beta2[:, :, :, 1:]) + CharbonnierFunc(m_Beta2[:, :, :-1, :] - m_Beta2[:, :, 1:, :])
            g_Occlusion = CharbonnierFunc(Occlusion[:, :, :, :-1] - Occlusion[:, :, :, 1:]) + CharbonnierFunc(Occlusion[:, :, :-1, :] - Occlusion[:, :, 1:, :])

            g_Spatial = g_Alpha1 + g_Beta1 + g_Alpha2 + g_Beta2

            return {'frame1': frame1, 'g_Spatial': g_Spatial, 'g_Occlusion': g_Occlusion} # 학습시에는 g_spatial, g_occlusion을 같이 내보냄. 
        else:
            return frame1 # 평가모드에서는 그냥 output만 제출
