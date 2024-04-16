import torch.nn as nn
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, x):
        return self.backbone(x)
    
    
class VitSimCLR(nn.Module):
    def __init__(self, out_dim, num_class):
        super(VitSimCLR, self).__init__()
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone.heads = nn.Linear(in_features=768, out_features=out_dim)
        self.desc_cls = nn.Linear(in_features=out_dim, out_features=num_class)

    def forward(self, x):
        global_feature = self.backbone(x)
        global_logit = self.desc_cls(global_feature)
        return global_feature, global_logit
