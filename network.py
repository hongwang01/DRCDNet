# This is code for dynamic RCDNet (DRCDNet), which is the extension of RCDNet published in CVPR2020 https://github.com/hongwang01/RCDNet
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as  F
from torch.autograd import Variable
import scipy.io as io
# initialized rain kernel dictionary D
rain_kernel = io.loadmat('init_kernel_dir.mat') ['C9'] # 3*64*9*9
# filtering on rainy image for initializing B^(0) and Z^(0),
filter = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9
filter = filter.unsqueeze(dim=0).unsqueeze(dim=0)
class DRCDNet(nn.Module):
    def __init__(self, args):
        super(DRCDNet, self).__init__()
        self.S  = args.stage                                                # Stage number S includes the initialization process
        self.iters = self.S -1                                              # not include the initialization process
        self.num_D = args.num_D                                             # No. rain kernel dictionary N=32
        self.num_M = args.num_M                                             # No. rain Map, d = 6
        self.num_ZB = args.num_ZB                                           # No. dual channel
        # Stepsize
        self.etaM = torch.Tensor([args.etaM])                               # stepsize initialization for rain map
        self.etaB = torch.Tensor([args.etaB])                               # stepsize initialization for Background
        self.etaAlpha = torch.Tensor([5])                                   # stepsize initialization for alpha
        self.eta1_S = self.make_eta(self.S, self.etaM)
        self.eta2_S = self.make_eta(self.S, self.etaB)
        self.eta3_S = self.make_eta(self.S, self.etaAlpha)
        self.eta_rcdb = self.make_eta(2,self.etaB)
        self.eta_rcdm = self.make_eta(1,self.etaM)
        # Rain kernel
        kernel = torch.FloatTensor(rain_kernel)
        self.C0 = nn.Parameter(data=kernel[:,:self.num_M,:,:], requires_grad=True)      # used in initialization process
        self.extend_C0 = nn.Parameter(data=kernel, requires_grad = True)
        self.C = nn.Parameter(data=kernel[:,:self.num_D,:,:], requires_grad=True)
        # filter for initializing B and Z
        self.C_z_const = filter.expand(self.num_ZB, 3, -1, -1)
        self.C_z = nn.Parameter(self.C_z_const, requires_grad=True)

        # proxNet Initiialize rain map of DRCDNet by adopting the initialization stage of RCDNet
        self.alphaNet0 = (nn.Linear(self.num_D*self.num_M, self.num_D*self.num_M))
        self.proxNet_B_0= Bnet(args)
        self.proxNet_B_S = self.make_Bnet(self.S, args)
        self.proxNet_M_init = self.make_Mnet(2, args)
        self.proxNet_B_init = self.make_Bnet(2, args)
        self.proxNet_M_S = self.make_Mnet(self.S, args)
        self.proxNet_alpha_S = self.make_Alphanet(self.S, args)
        self.proxNet_B_last_layer = Bnet(args)                                               # fine-tune at the last stage
        self.tau_const = torch.Tensor([1])
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)                          # for sparse rain layer
        # Transform the rain map with channel N=32 initialized by RCDNet and take the partial channel with channel d=6 as the initialized rain map in DRCDNet
        convert = torch.eye(self.num_M, self.num_M)
        big_convert = convert.unsqueeze(dim=2).unsqueeze(dim=3)
        self.convert_conv_layer1 = nn.Parameter(data=big_convert, requires_grad=True)
        self.convert_conv_layer2 = nn.Parameter(data=big_convert, requires_grad=True)

    def make_Bnet(self, iters, args):
        layers = []
        for i in range(iters):
            layers.append(Bnet(args))
        return nn.Sequential(*layers)

    def make_Mnet(self, iters, args):
        layers = []
        for i in range(iters):
            layers.append(Mnet(args))
        return nn.Sequential(*layers)

    def make_Alphanet(self, iters, args):
        layers = []
        for i in range(iters):
            layers.append(Alphanet(args))
        return nn.Sequential(*layers)

    def make_eta(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1)
        eta = nn.Parameter(data=const_f, requires_grad=True)
        return eta

    def forward(self, input):
        # save mid-updating results
        b, h, w = input.size()[0], input.size()[2], input.size()[3]
        ListB = []
        ListR = []
        ListR_rcd =[]
        ListB_rcd =[]
############################# Adopting the stage 0 and stage 1  of RCDNet to initize rain map, background for DRCDNet ##################################
        # initialize B0 and BZ0 (M0 = MZ0=0)
        Z00 = F.conv2d(input, self.C_z, stride=1, padding=1)
        input_ini = torch.cat((input, Z00), dim=1)
        BZ_ini = self.proxNet_B_0(input_ini)
        B0 = BZ_ini[:, :3, :, :]
        BZ0 = BZ_ini[:, 3:, :, :]

        # updating M0--->M1
        ES = input - B0
        ECM = F.relu(ES - self.tau)                                             # for sparse rain layer
        GM = F.conv_transpose2d(ECM, self.C0/10, stride=1, padding=4)           # /10 for controlling the updating speed
        M = self.proxNet_M_init[0](GM)
        CM = F.conv2d(M, self.C0 /10, stride =1, padding = 4)
        # Updating B0-->B1
        EB = input - CM
        EX = B0-EB
        GX = EX
        B1 = B0-self.eta_rcdb[0,:]/10*GX
        input_dual = torch.cat((B1, BZ0), dim=1)
        out_dual = self.proxNet_B_init[0](input_dual)
        B = out_dual[:,:3,:,:]
        BZ = out_dual[:,3:,:,:]
        ListB_rcd.append(B)
        ListR_rcd.append(CM)
       # M-net
        ES = input - B
        ECM = CM- ES
        GM = F.conv_transpose2d(ECM,  self.C0/10, stride =1, padding = 4)
        input_new = M - self.eta_rcdm[0,:]/10*GM
        M = self.proxNet_M_init[1](input_new)
        # B-net
        CM = F.conv2d(M, self.C0/10, stride =1, padding = 4)
        ListR_rcd.append(CM)
        EB = input - CM
        EX = B - EB
        GX = EX
        x_dual = B - self.eta_rcdb[1,:]/10*GX
        input_dual = torch.cat((x_dual,BZ), dim=1)
        out_dual  = self.proxNet_B_init[1](input_dual)
        B = out_dual[:,:3,:,:]                                          # This B is used as the B0 of DRCDNet
        BZ = out_dual[:,3:,:,:]                                        # This BZ is used as the Z0 of DRCDNet
        ListB_rcd.append(B)
        M1 = F.conv2d(F.relu(M),self.convert_conv_layer1, stride=1, padding=0)
        M = F.conv2d(M1, self.convert_conv_layer2, stride=1, padding=0)  # This M is used as the M0 of DRCDNet

  ######################################## Adopt the iterative process of DRCDNet to further initialize alpha, rain map, background  ##################
        #1st iteration：Updating B0, M0-->ALPHA0
        M_re = M.reshape(1, b*self.num_M, h, w)
        C_re = self.C.reshape(1, 3*self.num_D, 9, 9).expand(b * self.num_M,-1,-1,-1).reshape(b * self.num_M* 3*self.num_D, 1, 9,9)
        CM = F.conv2d(M_re, C_re / 10, groups=b * self.num_M, stride=1, padding=4)
        CM_re = CM.reshape(b, self.num_M, 3, self.num_D, h, w).permute(0, 1, 3, 2, 4, 5).reshape(b, self.num_M * self.num_D, 3 * h * w)
        CM_re_trans =  CM_re
        R_hat = input - B
        R_hat_re = R_hat.reshape(b, 3*h*w).unsqueeze(dim=2)
        Galpha = torch.bmm(CM_re_trans,R_hat_re).squeeze(dim=2)
        Galpha_re = Galpha.reshape(b, self.num_M, self.num_D)
        alpha = self.proxNet_alpha_S[0](Galpha_re)

        # 1st iteration：Updating B0，M0, ALPHA0-->M1
        C_per = self.C.permute(1, 0, 2, 3).reshape(self.num_D, -1)
        alphaC = torch.matmul(alpha, C_per / 10).reshape(b, self.num_M, 3, 9, 9).permute(0, 2, 1, 3, 4).reshape(b * 3,self.num_M, 9, 9)  # 80*243    80 = 16*5  num_M：5
        R = F.conv2d(M_re, alphaC, groups=b, stride=1, padding=4)
        R_re = R.reshape(b, 3, h, w)
        R_hat_cut = (R_hat-R_re).reshape(1, 3 * b, h, w)
        Epsilon = F.conv_transpose2d(R_hat_cut, alphaC, groups=b, stride=1, padding=4).reshape(b, self.num_M, h, w)  # /10 for controlling the updating speed
        Epsilon_re = Epsilon.reshape(b, self.num_M, h, w)
        GM = M + self.eta1_S[0,:]/10 * Epsilon_re
        M = self.proxNet_M_S[0](GM)

        # 1st iteration: Updating alpha0, B0, M1-->B1
        M_re = M.reshape(1, b * self.num_M, h, w)
        R = F.conv2d(M_re, alphaC, groups=b, stride=1, padding=4)
        R_re = R.reshape(b, 3, h, w)
        B_hat = input - R_re
        B_mid = (1-self.eta2_S[0]/10) * B + self.eta2_S[0,:]/10 * B_hat
        inputB_concat = torch.cat((B_mid, BZ), dim=1)
        B_dual = self.proxNet_B_S[0](inputB_concat)
        B = B_dual[:, :3, :, :]
        BZ = B_dual[:, 3:, :, :]
        ListB.append(B)
        ListR.append(R_re)
        C_re = self.C.reshape(1, 3 * self.num_D, 9, 9).expand(b * self.num_M, -1, -1, -1).reshape(b * self.num_M * 3 * self.num_D, 1, 9, 9)

        for i in range(self.iters):
            # Alpha-Net
            CM = F.conv2d(M_re, C_re/10, groups=b * self.num_M, stride=1, padding=4)
            CM_re = CM.reshape(b, self.num_M, 3, self.num_D, h, w).permute(0, 1, 3, 2, 4, 5).reshape(b, self.num_M * self.num_D, 3 * h * w)
            CM_re_trans = CM_re
            R = F.conv2d(M_re, alphaC, groups=b, stride=1, padding=4)  # /10 for controlling the updating speed
            R_re = R.reshape(b, 3, h, w)
            R_hat = input - B - R_re
            R_hat_re = R_hat.reshape(b, 3 * h * w).unsqueeze(dim=2)
            Galpha = torch.bmm(CM_re_trans, R_hat_re).squeeze(dim=2)
            Galpha_re = Galpha.reshape(b, self.num_M, self.num_D)
            alpha = self.proxNet_alpha_S[i+1](alpha + self.eta3_S[i+1, :] / (h * w * 50) * Galpha_re)
            ListAlpha.append(alpha)

            #M-net
            R_re =torch.bmm(alpha.reshape(b,1, self.num_M*self.num_D), CM_re).reshape(b, 3, h, w)
            alphaC = torch.matmul(alpha, C_per / 10).reshape(b, self.num_M, 3, 9, 9).permute(0, 2, 1, 3, 4).reshape(b * 3, self.num_M, 9, 9)
            R_hat_cut = (input - B - R_re).reshape(1, 3 * b, h, w)
            Epsilon = F.conv_transpose2d(R_hat_cut, alphaC, groups=b, stride=1, padding=4).reshape(b, self.num_M, h, w)  # /10 for controlling the updating speed
            Epsilon_re = Epsilon.reshape(b, self.num_M, h, w)
            GM = M + self.eta1_S[i+1, :] / 10 * Epsilon_re
            M = self.proxNet_M_S[i+1](GM)

            # B-net
            M_re = M.reshape(1, b * self.num_M, h, w)
            R = F.conv2d(M_re, alphaC, groups=b, stride=1, padding=4)
            R_re = R.reshape(b, 3, h, w)
            B_hat = input - R_re
            ListR.append(R_re)
            B_mid = (1 - self.eta2_S[i+1, :] / 10) * B + self.eta2_S[i+1, :] / 10 * B_hat
            inputB_concat = torch.cat((B_mid, BZ), dim=1)
            B_dual = self.proxNet_B_S[i + 1](inputB_concat)
            B = B_dual[:, :3, :, :]
            BZ = B_dual[:, 3:, :, :]
            ListB.append(B)
        BZ_adjust = self.proxNet_B_last_layer(B_dual)
        B = BZ_adjust[:, :3, :, :]
        ListB.append(B)
        return B0, ListB, ListR, ListB_rcd, ListR_rcd

class Mnet(nn.Module):
    def __init__(self, args):
        super(Mnet, self).__init__()
        self.channels = args.num_M
        self.T = args.T                                           # the number of resblocks in each proxNet
        self.layer = self.make_resblock(self.T)
        self.tau0 = torch.Tensor([args.Mtau])
        self.tau_const = self.tau0.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(-1,self.channels,-1,-1)
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)  # for sparse rain map
    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              nn.BatchNorm2d(self.channels),
                              nn.ReLU(),
                              nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              nn.BatchNorm2d(self.channels),
                          ))
        return nn.Sequential(*layers)
    def forward(self, input):
        M = input
        for i in range(self.T):
            M = F.relu(M+self.layer[i](M))
        M = F.relu(M-self.tau)
        return M

class Alphanet(nn.Module):
    def __init__(self, args):
        super(Alphanet, self).__init__()
        self.num_D = args.num_D
        self.num_M = args.num_M
        self.alphaT = args.alphaT
        self.layer = self.make_resblock(self.alphaT)
        self.tau0 = torch.Tensor([0.1])
        self.tau_const = self.tau0.unsqueeze(dim=0).unsqueeze(dim=0).expand(-1,self.num_M,self.num_D)
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)  # for sparse rain map
    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(nn.Sequential(
                nn.Linear(self.num_D * self.num_M, self.num_D * self.num_M),
		        nn.ReLU(),
                nn.Linear(self.num_D * self.num_M, self.num_D * self.num_M),
                ))
        return nn.Sequential(*layers)

    def forward(self, input):
        alpha = input
        b = alpha.size()[0]
        alpha = alpha.reshape(-1, self.num_D * self.num_M)
        for i in range(self.alphaT):
            alpha = F.relu(alpha + self.layer[i](alpha))
        alpha = alpha.reshape(b, self.num_M, self.num_D)
        norm = torch.norm(alpha,2,dim=2)
        norm_re = norm.unsqueeze(dim=2).expand(-1,-1,self.num_D)
        alpha = torch.div(alpha,norm_re+1e-6)
        return alpha

# proxNet_B
class Bnet(nn.Module):
    def __init__(self, args):
        super(Bnet, self).__init__()
        self.channels = args.num_ZB + 3                            # 3 means R,G,B channels for color image
        self.T = args.T
        self.layer = self.make_resblock(self.T)
    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(),
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(self.channels),
                ))
        return nn.Sequential(*layers)

    def forward(self, input):
        B = input
        for i in range(self.T):
            B = F.relu(B + self.layer[i](B))
        return B
