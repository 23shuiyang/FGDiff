import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import darmo
from PIL import Image

from models.head import Decoder, DecoderPKD, DecoderSemiPKD
import torch.nn.functional as F
import timm
def _remove_module(state_dict, index=9):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[index:]
        new_state_dict[name] = v
    return new_state_dict

class EEEAC2(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple", decoder='ours', is_f=False):
        super(EEEAC2, self).__init__()
        self.is_f = is_f
        self.num_channels = num_channels
        self.eeeac2 =  darmo.create_model("eeea_c2", num_classes=1000, pretrained=True, auxiliary=True)

        for param in self.eeeac2.parameters():
            param.requires_grad = train_enc

        del self.eeeac2.feature_mix_layer 
        del self.eeeac2.classifier 
        if decoder == 'PKD':
            self.head = DecoderPKD(channels=[960, 112, 40, 24, 16, 16], act="ReLU", output_size=output_size,
                                readout=readout)
        elif decoder == 'SemiPKD':
            self.head = DecoderSemiPKD(channels=[960, 112, 40, 24, 16, 16], act="ReLU", output_size=output_size,
                                   readout=readout)
        elif decoder == 'ours':
            #self.head = Decoder(channels=[960, 112, 40, 24, 16, 16], act="Swish", output_size=output_size, readout=readout)
            self.head = Decoder(channels=[960, 112, 40, 24, 16, 16], act="ReLU", output_size=output_size, readout=readout)

    def forward(self, x):
        x = self.eeeac2.first_conv(x)
        out1 = x
        for i in range(len(self.eeeac2.blocks)):
            x = self.eeeac2.blocks[i](x)
            if i == 3:
                out2 = x
            elif i == 6:
                out3 = x
            elif i == 14:
                out4 = x

        x = self.eeeac2.final_expand_layer(x)
        out5 = x
        f, x = self.head(out1, out2, out3, out4, out5)
        if self.is_f:
            return f, x
        else:
            return x
class MobileNetV3(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple", decoder='ours', is_f=False):
        super(MobileNetV3, self).__init__()
        self.is_f = is_f
        self.model = models.mobilenet_v3_large(pretrained=True).features
        if decoder == 'PKD':
            self.head = DecoderPKD(channels=[960, 112, 40, 24, 16, 16], act="ReLU", output_size=output_size,
                                readout=readout)
        elif decoder == 'SemiPKD':
            self.head = DecoderSemiPKD(channels=[960, 112, 40, 24, 16, 16], act="ReLU", output_size=output_size,
                                   readout=readout)
        elif decoder == 'ours':
            self.head = Decoder(channels=[960, 112, 40, 24, 16, 16], act="ReLU", output_size=output_size,
                                readout=readout)
    def forward(self, x):
        for i in range(len(self.model)):
            x = self.model[i](x)      
            if i == 0:
                out1 = x
            elif i == 3:
                out2 = x
            elif i == 6:
                out3 = x
            elif i == 12:
                out4 = x
        out5 = x
        f, x = self.head(out1, out2, out3, out4, out5)
        if self.is_f:
            return f, x
        else:
            return x

class EfficientNet(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple", decoder='ours', is_f=False):
        super(EfficientNet, self).__init__()
        pretrained_cfg = timm.models.create_model('tf_efficientnet_b0').default_cfg
        pretrained_cfg['file'] = r'.\checkpoints\tf_efficientnet_b0.ns_jft_in1k.bin'
        self.model = timm.create_model('tf_efficientnet_b0', pretrained=True, pretrained_cfg=pretrained_cfg)
        #self.model = timm.create_model('tf_efficientnet_b0', pretrained=True)
        self.is_f = is_f
        del self.model.global_pool
        del self.model.classifier
        if decoder == 'PKD':
            self.head = DecoderPKD(channels=[1280, 112, 40, 24, 32, 32], act="ReLU", output_size=output_size,
                                readout=readout)
        elif decoder == 'SemiPKD':
            self.head = DecoderSemiPKD(channels=[1280, 112, 40, 24, 32, 32], act="ReLU", output_size=output_size,
                                   readout=readout)
        elif decoder == 'ours':
            self.head = Decoder(channels=[1280, 112, 40, 24, 32, 32], act="ReLU", output_size=output_size,
                                readout=readout)


    def forward(self, x):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)  
        #del.act1(x)
        out1 = x
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x)            
            if i == 1:
                out2 = x
            elif i == 2:
                out3 = x
            elif i == 4:
                out4 = x
        x = self.model.conv_head(x)  
        x = self.model.bn2(x)  
        #x = self.model.act2(x)
        out5 = x
        f, x = self.head(out1, out2, out3, out4, out5)
        if self.is_f:
            return f, x
        else:
            return x

class EfficientNetB4(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple", decoder='ours', is_f=False):
        super(EfficientNetB4, self).__init__()
        pretrained_cfg = timm.models.create_model('tf_efficientnet_b4').default_cfg
        pretrained_cfg['file'] = r'.\checkpoints\tf_efficientnet_b4.ns_jft_in1k.bin'
        self.model = timm.create_model('tf_efficientnet_b4', pretrained=True, pretrained_cfg=pretrained_cfg)
        #self.model = timm.create_model('tf_efficientnet_b4', pretrained=True)
        self.is_f = is_f
        del self.model.global_pool
        del self.model.classifier
        if decoder == 'PKD':
            self.head = DecoderPKD(channels=[1792, 160, 56, 32, 48, 48], act="ReLU", output_size=output_size,
                                readout=readout)
        elif decoder == 'SemiPKD':
            self.head = DecoderSemiPKD(channels=[448, 160, 56, 32, 48, 48], act="ReLU", output_size=output_size,
                                   readout=readout)
        elif decoder == 'ours':
            self.head = Decoder(channels=[1792, 160, 56, 32, 48, 48], act="ReLU", output_size=output_size,
                                readout=readout)

    def forward(self, x):
        x = self.model.conv_stem(x)  
        x = self.model.bn1(x)  
       #x = self.model.act1(x)
        out1 = x
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x)            
            if i == 1:
                out2 = x
            elif i == 2:
                out3 = x
            elif i == 4:
                out4 = x
        x = self.model.conv_head(x)
        x = self.model.bn2(x)
        #x = self.model.act2(x)
        out5 = x
        f, x = self.head(out1, out2, out3, out4, out5)
        if self.is_f:
            return f, x
        else:
            return x


class EfficientNetB7(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple", decoder='ours', is_f=False):
        super(EfficientNetB7, self).__init__()
        pretrained_cfg = timm.models.create_model('tf_efficientnet_b7').default_cfg
        pretrained_cfg['file'] = r'.\checkpoints\tf_efficientnet_b7.ns_jft_in1k.bin'
        self.model = timm.create_model('tf_efficientnet_b7', pretrained=True, pretrained_cfg=pretrained_cfg)
        #self.model = timm.create_model('tf_efficientnet_b7', pretrained=True)
        self.is_f = is_f
        del self.model.global_pool
        del self.model.classifier
        if decoder == 'PKD':
            self.head = DecoderPKD(channels=[2560, 224, 80, 48, 64, 64], act="ReLU", output_size=output_size,
                                readout=readout)
        elif decoder == 'SemiPKD':
            self.head = DecoderSemiPKD(channels=[2560, 224, 80, 48, 64, 64], act="ReLU", output_size=output_size,
                                   readout=readout)
        elif decoder == 'ours':
            self.head = Decoder(channels=[2560, 224, 80, 48, 64, 64], act="ReLU", output_size=output_size,
                                readout=readout)
    def forward(self, x):
        x = self.model.conv_stem(x)  
        x = self.model.bn1(x)  
        #x = self.model.act1(x)
        out1 = x
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x)            
            if i == 1:
                out2 = x
            elif i == 2:
                out3 = x
            elif i == 4:
                out4 = x
        x = self.model.conv_head(x)  
        x = self.model.bn2(x)  
        #x = self.model.act2(x)
        out5 = x
        f, x = self.head(out1, out2, out3, out4, out5)
        if self.is_f:
            return f, x
        else:
            return x

class MobileNetV3_21k(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple", decoder='ours', is_f=False):
        super(MobileNetV3_21k, self).__init__()

        self.model = timm.create_model('mobilenetv3_large_100_miil_in21k', pretrained=True)
        self.is_f = is_f
        del self.model.global_pool
        del self.model.conv_head
        del self.model.act2
        del self.model.flatten
        del self.model.classifier

        self.stem = nn.Sequential(
            self.model.conv_stem,
            self.model.bn1,
            self.model.act1,
        )
        if decoder == 'PKD':
            self.head = DecoderPKD(channels=[960, 112, 40, 24, 16, 16], act="ReLU", output_size=output_size,
                                readout=readout)
        elif decoder == 'SemiPKD':
            self.head = DecoderSemiPKD(channels=[960, 112, 40, 24, 16, 16], act="ReLU", output_size=output_size,
                                   readout=readout)
        elif decoder == 'ours':
            self.head = Decoder(channels=[960, 112, 40, 24, 16, 16], act="ReLU", output_size=output_size,
                                readout=readout)
    def forward(self, x):
        x = self.stem(x)
        out1 = x
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x) 
            if i == 1:
                out2 = x
            elif i == 2:
                out3 = x
            elif i == 4:
                out4 = x
            elif i == 6:
                out5 = x
        f, x = self.head(out1, out2, out3, out4, out5)
        if self.is_f:
            return f, x
        else:
            return x

class MobileNetV3_1k(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple", decoder='ours',is_f=False):
        super(MobileNetV3_1k, self).__init__()

        pretrained_cfg = timm.models.create_model('mobilenetv3_large_100_miil').default_cfg
        pretrained_cfg['file'] = r'.\checkpoints\mobilenetv3_large_100.miil_in21k_ft_in1k.bin'
        self.model = timm.create_model('mobilenetv3_large_100_miil', pretrained=True, pretrained_cfg=pretrained_cfg)

        del self.model.global_pool
        del self.model.conv_head
        del self.model.act2
        del self.model.flatten
        del self.model.classifier
        self.is_f = is_f
        self.stem = nn.Sequential(
            self.model.conv_stem,
            self.model.bn1,
        )
        if decoder == 'PKD':
            self.head = DecoderPKD(channels=[960, 112, 40, 24, 16, 16], act="ReLU", output_size=output_size,
                                readout=readout)
        elif decoder == 'SemiPKD':
            self.head = DecoderSemiPKD(channels=[960, 112, 40, 24, 16, 16], act="ReLU", output_size=output_size,
                                   readout=readout)
        elif decoder == 'ours':
            self.head = Decoder(channels=[960, 112, 40, 24, 16, 16], act="ReLU", output_size=output_size,
                                readout=readout)

    def forward(self, x):
        x = self.stem(x)
        out1 = x
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x) 
            if i == 1:
                out2 = x
            elif i == 2:
                out3 = x
            elif i == 4:
                out4 = x
            elif i == 6:
                out5 = x
        f, x = self.head(out1, out2, out3, out4, out5)
        if self.is_f:
            return f, x
        else:
            return x

class OFA595(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480,640), readout="simple", decoder='ours', is_f=False):
        super(OFA595, self).__init__()
        self.is_f = is_f
        self.num_channels = num_channels
        self.model =  darmo.create_model("ofa595_1k", num_classes=1000, pretrained=True, auxiliary=True)

        for param in self.model.parameters():
            param.requires_grad = train_enc

        del self.model.feature_mix_layer 
        del self.model.classifier 
        if decoder == 'PKD':
            self.head = DecoderPKD(channels=[1152, 136, 48, 32, 24, 24], act="ReLU", output_size=output_size,
                                readout=readout)
        elif decoder == 'SemiPKD':
            self.head = DecoderSemiPKD(channels=[1152, 136, 48, 32, 24, 24], act="ReLU", output_size=output_size,
                                   readout=readout)
        elif decoder == 'ours':
            self.head = Decoder(channels=[1152, 136, 48, 32, 24, 24], act="ReLU", output_size=output_size,
                                readout=readout)

    def forward(self, x):
        x = self.model.first_conv(x)
        out1 = x
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x)
            if i == 3:
                out2 = x
            elif i == 7:
                out3 = x
            elif i == 15:
                out4 = x

        x = self.model.final_expand_layer(x)
        out5 = x
        f, x = self.head(out1, out2, out3, out4, out5)
        if self.is_f:
            return f, x
        else:
            return x

class RepViT_m2_3(nn.Module):
    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480, 640), readout="simple", decoder='ours', is_f=False):
        super(RepViT_m2_3, self).__init__()
        pretrained_cfg = timm.models.create_model('repvit_m2_3').default_cfg
        pretrained_cfg['file'] = r'.\checkpoints\repvit_m2_3.dist_450e_in1k.bin'
        self.model = timm.create_model('repvit_m2_3', pretrained=True, pretrained_cfg=pretrained_cfg)
        self.is_f = is_f
        del self.model.head_drop
        del self.model.head
        if decoder == 'PKD':
            self.head = DecoderPKD(channels=[640, 320, 160, 80, 20, 20], act="ReLU", output_size=output_size,
                                readout=readout)
        elif decoder == 'SemiPKD':
            self.head = DecoderSemiPKD(channels=[640, 320, 160, 80, 20, 20], act="ReLU", output_size=output_size,
                                   readout=readout)
        elif decoder == 'ours':
            self.head = Decoder(channels=[640, 320, 160, 80, 20, 20], act="ReLU", output_size=output_size,
                                readout=readout)
        self.pixel_shuffle = nn.PixelShuffle(2)
    def forward(self, x):
        x = self.model.stem(x)
        out1 = self.pixel_shuffle(x)
        for i in range(len(self.model.stages)):
            x = self.model.stages[i](x)
            if i == 0:
                out2 = x
            elif i == 1:
                out3 = x
            elif i == 2:
                out4 = x
            elif i == 3:
                out5 = x
        f, x = self.head(out1, out2, out3, out4, out5)
        if self.is_f:
            return f, x
        else:
            return x
class RepViT_m1_5(nn.Module):
    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480, 640), readout="simple", decoder='ours', is_f=False):
        super(RepViT_m1_5, self).__init__()
        pretrained_cfg = timm.models.create_model('repvit_m1_5').default_cfg
        pretrained_cfg['file'] = r'.\checkpoints\repvit_m1_5.dist_450e_in1k.bin'
        self.model = timm.create_model('repvit_m1_5', pretrained=True, pretrained_cfg=pretrained_cfg)
        self.is_f = is_f
        del self.model.head_drop
        del self.model.head
        if decoder == 'PKD':
            self.head = DecoderPKD(channels=[512, 256, 128, 64, 16, 16], act="ReLU", output_size=output_size,
                                readout=readout)
        elif decoder == 'SemiPKD':
            self.head = DecoderSemiPKD(channels=[512, 256, 128, 64, 16, 16], act="ReLU", output_size=output_size,
                                   readout=readout)
        elif decoder == 'ours':
            self.head = Decoder(channels=[512, 256, 128, 64, 16, 16], act="ReLU", output_size=output_size,
                                readout=readout)
        self.pixel_shuffle = nn.PixelShuffle(2)
    def forward(self, x):
        x = self.model.stem(x)
        out1 = self.pixel_shuffle(x)
        for i in range(len(self.model.stages)):
            x = self.model.stages[i](x)
            if i == 0:
                out2 = x
            elif i == 1:
                out3 = x
            elif i == 2:
                out4 = x
            elif i == 3:
                out5 = x
        f, x = self.head(out1, out2, out3, out4, out5)
        if self.is_f:
            return f, x
        else:
            return x

class RepViT_m0_9(nn.Module):
    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480, 640), readout="simple", decoder='ours', is_f=False):
        super(RepViT_m0_9, self).__init__()
        pretrained_cfg = timm.models.create_model('repvit_m0_9').default_cfg
        pretrained_cfg['file'] = r'.\checkpoints\repvit_m0_9.dist_450e_in1k.bin'
        self.model = timm.create_model('repvit_m0_9', pretrained=True, pretrained_cfg=pretrained_cfg)
        self.is_f = is_f
        del self.model.head_drop
        del self.model.head
        if decoder == 'PKD':
            self.head = DecoderPKD(channels=[384, 192, 96, 48, 12, 12], act="ReLU", output_size=output_size,
                                readout=readout)
        elif decoder == 'SemiPKD':
            self.head = DecoderSemiPKD(channels=[384, 192, 96, 48, 12, 12], act="ReLU", output_size=output_size,
                                   readout=readout)
        elif decoder == 'ours':
            self.head = Decoder(channels=[384, 192, 96, 48, 12, 12], act="ReLU", output_size=output_size,
                                readout=readout)
        self.pixel_shuffle = nn.PixelShuffle(2)
    def forward(self, x):
        x = self.model.stem(x)
        out1 = self.pixel_shuffle(x)
        for i in range(len(self.model.stages)):
            x = self.model.stages[i](x)
            if i == 0:
                out2 = x
            elif i == 1:
                out3 = x
            elif i == 2:
                out4 = x
            elif i == 3:
                out5 = x
        f, x = self.head(out1, out2, out3, out4, out5)
        if self.is_f:
            return f, x
        else:
            return x


class EfficientFormerV2_S2(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1, output_size=(480, 640), readout="simple",
                 decoder='ours', is_f=False):
        super(EfficientFormerV2_S2, self).__init__()

        pretrained_cfg = timm.create_model('efficientformerv2_l').default_cfg
        pretrained_cfg['file'] = r'.\checkpoints\efficientformerv2_l.snap_dist_in1k.bin'
        self.model = timm.create_model('efficientformerv2_l', pretrained=True, pretrained_cfg=pretrained_cfg)
        self.is_f = is_f

        self.feature_info = self.model.feature_info

        del self.model.global_pool


        features_channels = []
        for feat in self.feature_info:
            features_channels.append(feat['num_chs'])

        if decoder == 'PKD':
            self.head = DecoderPKD(channels=features_channels, act="ReLU", output_size=output_size,
                                   readout=readout)
        elif decoder == 'SemiPKD':
            self.head = DecoderSemiPKD(channels=features_channels, act="ReLU", output_size=output_size,
                                       readout=readout)
        elif decoder == 'ours':
            self.head = Decoder(channels=[384, 192, 80, 40, 40, 40], act="ReLU", output_size=output_size,
                                readout=readout)

    def forward(self, x):
        out1 = self.model.stem(x)
        out2 = self.model.stages[0](out1)
        out3 = self.model.stages[1](out2)
        out4 = self.model.stages[2](out3)
        out5 = self.model.stages[3](out4)
        out1 = F.interpolate(out1, size=(out1.shape[2]*2, out1.shape[3]*2), mode='bilinear', align_corners=False)
        f, x = self.head(out1, out2, out3, out4, out5)
        if self.is_f:
            return f, x
        else:
            return x