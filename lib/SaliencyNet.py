import kornia
import timm
import torch
import torch.nn as nn
import darmo
import torch.nn.functional as F

def list_sum(x):
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

def _make_output(input, output, act="ReLU", upsampling=True):
    linear_upsampling = nn.UpsamplingNearest2d(scale_factor=2)
    if upsampling:
        return nn.Sequential(
            nn.Conv2d(input, out_channels = output, kernel_size = 3, padding = 1, bias = False),
            nn.ReLU(inplace=True) if act == "ReLU" else MemoryEfficientSwish(),
            linear_upsampling,
        )
    else:
        if act != 'Sigmoid':
            return nn.Sequential(
                nn.Conv2d(input, out_channels = output, kernel_size = 3, padding = 1, bias = False),
                nn.ReLU(inplace=True) if act == "ReLU" else MemoryEfficientSwish(),
            )
        elif act == 'Sigmoid':
            return nn.Sequential(
                nn.Conv2d(input, out_channels = output, kernel_size = 3, padding = 1, bias = False),
                nn.Sigmoid(),
            )
class EEEAC2(nn.Module):
    def __init__(self, train_enc=False, act='ReLU'):
        super(EEEAC2, self).__init__()
        self.eeeac2 =  darmo.create_model("eeea_c2", num_classes=1000, pretrained=True, auxiliary=True)
        for param in self.eeeac2.parameters():
            param.requires_grad = train_enc
        del self.eeeac2.feature_mix_layer
        del self.eeeac2.classifier
        del self.eeeac2.final_expand_layer
        channels = [112, 40, 24, 16, 16]
        upsampling = [True, True, True, True, False, False]
        self.deconv_layer0 = _make_output(input=channels[0], output=channels[1], act=act, upsampling=upsampling[0])
        self.deconv_layer1 = _make_output(input=channels[1], output=channels[2], act=act, upsampling=upsampling[1])
        self.deconv_layer2 = _make_output(input=channels[2], output=channels[3], act=act, upsampling=upsampling[2])
        self.deconv_layer3 = _make_output(input=channels[3], output=channels[4], act=act, upsampling=upsampling[3])
        self.deconv_layer4 = _make_output(input=channels[4], output=channels[4], act=act, upsampling=upsampling[4])
        self.deconv_layer5 = _make_output(input=channels[4], output=1, act='Sigmoid', upsampling=upsampling[5])
        self.blur = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))

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

        # x = self.eeeac2.final_expand_layer(x)
        # out5 = x
        x0 = self.deconv_layer0(out4)

        x1 = list_sum([x0, out3])
        x1 = self.deconv_layer1(x1)

        x2 = list_sum([x1, out2])
        x2 = self.deconv_layer2(x2)

        x3 = list_sum([x2, out1])
        x3 = self.deconv_layer3(x3)

        x = self.deconv_layer4(x3)
        x = self.deconv_layer5(x)

        x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
        x = self.blur(x)
        return x

    def forward_encode(self, x):
        return self.forward(x)

class EfficientNetB0(nn.Module):

    def __init__(self, train_enc=False, act='ReLU'):
        super(EfficientNetB0, self).__init__()
        pretrained_cfg = timm.models.create_model('tf_efficientnet_b0').default_cfg
        pretrained_cfg['file'] = './ckpt/tf_efficientnet_b0.ns_jft_in1k.bin'
        self.model = timm.create_model('tf_efficientnet_b0', pretrained=True, pretrained_cfg=pretrained_cfg)
        del self.model.global_pool
        del self.model.classifier
        channels = [112, 40, 24, 32, 32]
        upsampling = [True, True, True, True, False, False]
        self.deconv_layer0 = _make_output(input=channels[0], output=channels[1], act=act, upsampling=upsampling[0])
        self.deconv_layer1 = _make_output(input=channels[1], output=channels[2], act=act, upsampling=upsampling[1])
        self.deconv_layer2 = _make_output(input=channels[2], output=channels[3], act=act, upsampling=upsampling[2])
        self.deconv_layer3 = _make_output(input=channels[3], output=channels[4], act=act, upsampling=upsampling[3])
        self.deconv_layer4 = _make_output(input=channels[4], output=channels[4], act=act, upsampling=upsampling[4])
        self.deconv_layer5 = _make_output(input=channels[4], output=1, act='Sigmoid', upsampling=upsampling[5])
        self.blur = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))

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

        x0 = self.deconv_layer0(out4)

        x1 = list_sum([x0, out3])
        x1 = self.deconv_layer1(x1)

        x2 = list_sum([x1, out2])
        x2 = self.deconv_layer2(x2)

        x3 = list_sum([x2, out1])
        x3 = self.deconv_layer3(x3)

        x = self.deconv_layer4(x3)
        x = self.deconv_layer5(x)

        x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
        x = self.blur(x)

        return x

    def forward_encode(self, x):
        return self.forward(x)