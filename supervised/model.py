import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TransformerModel(nn.Module):
    def __init__(self, num_class, args):
        super(TransformerModel, self).__init__()
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone.heads = nn.Linear(in_features=768, out_features=2048)
        if args.softmax:
            self.desc_cls = nn.Linear(in_features=2048, out_features=num_class)
        else:
            self.arcface_cls = ArcMarginProduct(in_features=2048, out_features=num_class, s=args.s, m=args.m)

    def forward(self, x, target, args):
        global_feature = self.backbone(x)
        if args.softmax:
            global_logit = self.desc_cls(global_feature)
        elif not args.softmax and self.training:
            global_logit = self.arcface_cls(global_feature, target)
        else:
            global_logit = None
        return global_feature, global_logit


class TransformerLargeModel(nn.Module):
    def __init__(self, num_class, args):
        super(TransformerLargeModel, self).__init__()
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone.heads = nn.Linear(in_features=768, out_features=2048)
        if args.softmax:
            self.desc_cls = nn.Linear(in_features=2048, out_features=num_class)
        else:
            self.arcface_cls = ArcMarginProduct(in_features=2048, out_features=num_class, s=args.s, m=args.m)

    def forward(self, x, target, args):
        global_feature = self.backbone(x)
        if args.softmax:
            global_logit = self.desc_cls(global_feature)
        elif not args.softmax and self.training:
            global_logit = self.arcface_cls(global_feature, target)
        else:
            global_logit = None
        return global_feature, global_logit


class Model(nn.Module):
    def __init__(self, num_class, args):
        super(Model, self).__init__()
        # set up backbone
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.ModuleList()
        module_list = []
        for name, module in backbone.named_children():
            module_list.append(module)
            if name[:5] == "layer":
                self.backbone.append(nn.Sequential(*module_list))
                module_list = []
            if name == "layer4":
                break
        self.global_pool = GeneralizedMeanPoolingP(norm=3.0)
        if args.softmax:
            self.desc_cls = nn.Linear(in_features=2048, out_features=num_class)
        else:
            self.arcface_cls = ArcMarginProduct(in_features=2048, out_features=num_class, s=args.s, m=args.m)

    def forward(self, x, target, args):
        x1 = self.backbone[0](x)
        x2 = self.backbone[1](x1)
        x3 = self.backbone[2](x2)
        x4 = self.backbone[3](x3)
        global_feature = self.global_pool(x4).squeeze()
        if args.softmax:
            global_logit = self.desc_cls(global_feature)
        elif not args.softmax and self.training:
            global_logit = self.arcface_cls(global_feature, target)
        else:
            global_logit = None
        return global_feature, global_logit


class Model101(nn.Module):
    def __init__(self, num_class, args):
        super(Model101, self).__init__()
        # set up backbone
        backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.backbone = nn.ModuleList()
        module_list = []
        for name, module in backbone.named_children():
            module_list.append(module)
            if name[:5] == "layer":
                self.backbone.append(nn.Sequential(*module_list))
                module_list = []
            if name == "layer4":
                break
        self.global_pool = GeneralizedMeanPoolingP(norm=3.0)
        if args.softmax:
            self.desc_cls = nn.Linear(in_features=2048, out_features=num_class)
        else:
            self.arcface_cls = ArcMarginProduct(in_features=2048, out_features=num_class, s=args.s, m=args.m)

    def forward(self, x, target, args):
        x1 = self.backbone[0](x)
        x2 = self.backbone[1](x1)
        x3 = self.backbone[2](x2)
        x4 = self.backbone[3](x3)
        global_feature = self.global_pool(x4).squeeze()
        if args.softmax:
            global_logit = self.desc_cls(global_feature)
        elif not args.softmax and self.training:
            global_logit = self.arcface_cls(global_feature, target)
        else:
            global_logit = None
        return global_feature, global_logit


class DOLG(nn.Module):
    def __init__(self, num_class, args):
        super(DOLG, self).__init__()
        # set up backbone
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.ModuleList()
        module_list = []
        for name, module in backbone.named_children():
            module_list.append(module)
            if name[:5] == "layer":
                self.backbone.append(nn.Sequential(*module_list))
                module_list = []
            if name == "layer4":
                break
        self.global_pool = GeneralizedMeanPoolingP(norm=3.0)
        self.local_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_t = nn.Linear(2048, 1024, bias=True)
        self.fc = nn.Linear(2048, 2048, bias=True)

        if args.softmax:
            self.desc_cls = nn.Linear(in_features=2048, out_features=num_class)
        else:
            self.arcface_cls = ArcMarginProduct(in_features=2048, out_features=num_class, s=args.s, m=args.m)

    def forward(self, x, target, args):
        x1 = self.backbone[0](x)
        x2 = self.backbone[1](x1)
        x3 = self.backbone[2](x2)
        x4 = self.backbone[3](x3)

        fg_o = self.global_pool(x4)
        fg_o = fg_o.view(fg_o.size(0), -1)

        fg = self.fc_t(fg_o)
        fg_norm = torch.norm(fg, p=2, dim=1)

        proj = torch.bmm(fg.unsqueeze(1), torch.flatten(x3, start_dim=2))
        proj = torch.bmm(fg.unsqueeze(2), proj).view(x3.size())
        proj = proj / (fg_norm * fg_norm).view(-1, 1, 1, 1)
        orth_comp = x3 - proj

        fo = self.local_pool(orth_comp)
        fo = fo.view(fo.size(0), -1)

        final_feat = torch.cat((fg, fo), 1)
        global_feature = self.fc(final_feat)

        if args.softmax:
            global_logit = self.desc_cls(global_feature)
        elif not args.softmax and self.training:
            global_logit = self.arcface_cls(global_feature, target)
        else:
            global_logit = None

        return global_feature, global_logit


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.15, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size()).to(input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output
