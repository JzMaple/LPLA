import math
import torch
import torch.nn as nn
import torchvision.models as models

from pygmtools.linear_solvers import sinkhorn, hungarian
from supervised.model import GeneralizedMeanPoolingP
from match.linear_attention import *
from match.perspective_transforms import *


class MatchModel(nn.Module):
    def __init__(self, num_class, args):
        super(MatchModel, self).__init__()
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

        # set up match module
        # self.transformer = TransformerModel(args)
        self.match_solver = MatchSolverModel(args)

        # set up heads
        # self.global_pool = nn.AdaptiveAvgPool1d(1)
        # self.desc_gcls = nn.Linear(in_features=args.local_trans_dim, out_features=num_class)

    def extract_local_feat(self, x):
        x1 = self.backbone[0](x)
        x2 = self.backbone[1](x1)
        x3 = self.backbone[2](x2)
        x4 = self.backbone[3](x3)
        return x3, x4

    def forward(self, im1, im2, M, labels):
        _, local_cnn_feat1 = self.extract_local_feat(im1)
        _, local_cnn_feat2 = self.extract_local_feat(im2)
        # local_cnn_feat3, _= self.extract_local_feat(im3)

        local_cnn_feat1 = F.normalize(local_cnn_feat1, p=2, dim=1)
        local_cnn_feat2 = F.normalize(local_cnn_feat2, p=2, dim=1)

        # local_trans_feat1, local_trans_feat2 = self.transformer(local_cnn_feat1, local_cnn_feat2)
        pos_mat, pos_mat_dis, pos_gt_mat, pos_nf = self.match_solver(
            local_cnn_feat1, local_cnn_feat2, M, None, pair="pos"
        )

        # global_feature = self.global_pool(pos_nf.permute(0, 2, 1)).squeeze()
        # global_logit = self.desc_gcls(global_feature)

        # return global_logit, pos_mat, pos_mat_dis, pos_gt_mat
        return pos_mat, pos_mat_dis, pos_gt_mat


class TransformerModel(torch.nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()
        self.fc = nn.Linear(args.local_cnn_dim, args.local_trans_dim, bias=True)
        self.local_feature_transformer = LocalFeatureTransformer()

    def forward(self, feat1, feat2):
        B, _, H, W = feat1.size()
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        feat1 = self.fc(feat1.permute(0, 2, 3, 1))
        feat2 = self.fc(feat2.permute(0, 2, 3, 1))

        feat1 = feat1.reshape(B, H * W, -1)
        feat2 = feat2.reshape(B, H * W, -1)

        # local feature transformer: self & cross
        feat1, feat2 = self.local_feature_transformer(feat1, feat2)
        feat1 = feat1.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        feat2 = feat2.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return feat1, feat2


class MatchSolverModel(torch.nn.Module):
    def __init__(self, args):
        super(MatchSolverModel, self).__init__()
        # bin_score = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
        # self.register_parameter("bin_score", bin_score)

    @staticmethod
    def generate_gt_mat(feat1, feat2, indices1, pair="pos"):
        B = feat1.shape[0]
        N1 = feat1.shape[1]
        N2 = feat2.shape[1]
        if pair == "pos":
            gt_mat = torch.zeros(B, N1, N2).cuda().scatter_(2, indices1.unsqueeze(-1), torch.ones(B, N1, 1).cuda())
            mask = (torch.sum(feat2, dim=-1) != 0).reshape(B, 1, N2).repeat(1, N1, 1)
            gt_mat = gt_mat * mask
        elif pair == "neg":
            gt_mat = torch.zeros(B, N1, N2).cuda()
        else:
            raise NotImplementedError
        return gt_mat.detach()

    def forward(self, feat1, feat2, M, saliency, pair="pos"):
        device = feat1.device
        B, C, H, W = feat1.size()

        if pair == "pos":
            # align feat2 to feat1
            feat2 = calculate_grid_sample(feat2, M, feat1.size(), mode="bilinear")
            feat1 = feat1.permute(0, 2, 3, 1).reshape(B, H * W, -1)
            feat2 = feat2.permute(0, 2, 3, 1).reshape(B, H * W, -1)
            # construct gt mat
            N = H * W
            mask = (torch.sum(feat2, dim=2) != 0).float()
            gt_mat = torch.diag_embed(mask).detach()
            # construct matching score
            matching_matrix = torch.eye(C).reshape(1, C, C).repeat(B, 1, 1).to(device)
            matching_matrix = torch.bmm(torch.bmm(feat1, matching_matrix), feat2.transpose(1, 2))
        elif pair == "neg":
            feat1 = feat1.permute(0, 2, 3, 1).reshape(B, H * W, -1)
            feat2 = feat2.permute(0, 2, 3, 1).reshape(B, H * W, -1)
            N = H * W
            gt_mat = torch.zeros((B, N, N)).cuda()
            matching_matrix = torch.eye(C).reshape(1, C, C).repeat(B, 1, 1).to(device)
            matching_matrix = torch.bmm(torch.bmm(feat1, matching_matrix), feat2.transpose(1, 2))
        else:
            raise NotImplementedError

        mat = sinkhorn(matching_matrix)
        mat_dis = hungarian(matching_matrix.detach())

        return mat, mat_dis, gt_mat, feat1


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self):
        super(LocalFeatureTransformer, self).__init__()

        self.d_model = 128
        self.nhead = 8
        self.layer_names = ["self"] * 4
        self.layers = nn.ModuleList([
            LoFTREncoderLayer(self.d_model, self.nhead, "linear")
            for _ in range(len(self.layer_names))
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == "self":
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == "cross":
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention="linear"):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == "linear" else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / d_model // 2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]
