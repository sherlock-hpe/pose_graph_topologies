# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
from mmengine.dist import get_dist_info
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.codecs.utils import get_simcc_normalized
from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.utils.rtmcc_block_with_gcn import RTMCCBlockWithGCN, ScaleNorm
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ..base_head import BaseHead
from mmpose.models.utils.gcn_two import GCNLayerTwo


OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class RTMCCHeadGMSWithGG(BaseHead):
    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 1,
        gau_cfg: ConfigType = dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='ReLU',
            use_rel_bias=False,
            pos_enc=False),
        loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
        data_type='coco',
    ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio
        self.data_type = data_type

        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if isinstance(in_channels, (tuple, list)):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

        # Define SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        self.final_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2)

        self.final_layer_1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2)

        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False))

        self.mlp_1 = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False))

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        self.gau = RTMCCBlockWithGCN(
            self.out_channels,
            gau_cfg['hidden_dims'],
            gau_cfg['hidden_dims'],
            s=gau_cfg['s'],
            expansion_factor=gau_cfg['expansion_factor'],
            dropout_rate=gau_cfg['dropout_rate'],
            drop_path=gau_cfg['drop_path'],
            attn_type='self-attn',
            act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'],
            pos_enc=gau_cfg['pos_enc'])

        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)
        #fusion
        self.gcn = GCNLayerTwo(self.data_type)
        # self.gcn_1 = GCNLayerTwo(self.data_type)
        self.linear = nn.Linear(gau_cfg['hidden_dims'], 4*flatten_dims, bias=False)
        self.conv_4_3_channels_align = nn.Conv2d(self.out_channels, self.in_channels//2,1)
        self.conv_3_2 = nn.Conv2d(self.in_channels, self.in_channels//2,1)
        self.conv_3_2_channels_align = nn.Conv2d(self.in_channels//2, self.in_channels // 4, 1)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // 8,1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.in_channels//8, self.in_channels, 1, bias=False),
            nn.Sigmoid()
            )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.conv1_to_1 = nn.Conv2d(self.in_channels//8, self.in_channels//4, 3, 2, 1)
        self.ca_1 = nn.Sequential(
            nn.Conv2d(self.in_channels//8, self.in_channels // 64, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.in_channels // 64, self.in_channels//8, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa_1 = nn.Sequential(
            nn.Conv2d(2, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.conv2_to_3 = nn.Conv2d(self.in_channels//4, self.in_channels//2,3,2,1)
        self.conv3_to_4 = nn.Conv2d(self.in_channels//2, self.in_channels,3,2,1)
        self.weight1 = torch.nn.Parameter(torch.tensor(0.5))
        self.weight2 = torch.nn.Parameter(torch.tensor(0.5))
        self.weight3 = torch.nn.Parameter(torch.tensor(0.5))
        self.weight4 = torch.nn.Parameter(torch.tensor(0.5))
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()



    def gcn_based_multi_scale_feature_fusion(self, feats: Tuple[Tensor]):
        inputs_stage_1 = feats[-4]
        inputs_stage_2 = feats[-3]
        inputs_stage_3 = feats[-2]
        inputs_stage_4 = feats[-1]

        batch_size, _, _, _ = inputs_stage_4.shape

        #gcn [B, self.in_channels, H, W]->[B, self.in_channels//2, 2*H, 2*W]
        inputs_stage_4 = self.final_layer_1(inputs_stage_4)
        inputs_stage_4 = torch.flatten(inputs_stage_4, 2)
        inputs_stage_4 = self.mlp_1(inputs_stage_4)
        inputs_stage_4 = self.gcn(inputs_stage_4)
        outputs_gcn = inputs_stage_4
        inputs_stage_4 = self.linear(inputs_stage_4)
        inputs_stage_4 = inputs_stage_4.view(batch_size, self.out_channels, 2*self.in_featuremap_size[1], 2*self.in_featuremap_size[0]).contiguous()
        inputs_stage_4 = self.conv_4_3_channels_align(inputs_stage_4)

        #channel attn [B, self.in_channels//2, 2*H, 2*W]->[B, self.in_channels, 2*H, 2*W]
        feats4_3 = torch.cat([inputs_stage_4,inputs_stage_3], dim =1)
        w4_3 = self.avg_pool(feats4_3)
        feats4_3_se = self.ca(w4_3)
        feats4_3 = feats4_3_se*feats4_3
        #spatial attn
        avg_out = torch.mean(feats4_3, dim=1, keepdim=True)
        max_out, _ = torch.max(feats4_3, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        sa = self.sa(spatial)
        feats4_3 = feats4_3*sa

        #upsampling stage3->stage2
        #[B, self.in_channels, 2*H, 2*W] -> [B, self.in_channels//4, 4*H, 4*W]
        feats3_2 = torch.nn.functional.interpolate(feats4_3, scale_factor=2, mode='bilinear', align_corners=False)
        feats3_2 = self.conv_3_2(feats3_2)
        feats3_2 = self.act1(feats3_2)
        feats3_2 = self.conv_3_2_channels_align(feats3_2)

        #stage1_to_stage2_downsampling_fusion
        #stage1 downsampling
        #channel attn
        stage_1 =  self.avg_pool(inputs_stage_1)
        inputs_stage_1 = self.ca_1(stage_1)*inputs_stage_1
        #spatial attn
        avg_out_1 = torch.mean(inputs_stage_1, dim=1, keepdim=True)
        max_out_1, _ = torch.max(inputs_stage_1, dim=1, keepdim=True)
        spatial_1 = torch.cat([avg_out_1, max_out_1], dim=1)
        inputs_stage_1 = self.sa_1(spatial_1)*inputs_stage_1
        inputs_stage_1 = self.conv1_to_1(inputs_stage_1)
        #fusion
        normed_weight_1 = torch.softmax(torch.stack([self.weight1, self.weight2]), dim=0)
        weight1, weight2 = normed_weight_1[0], normed_weight_1[1]
        inputs_stage_2 = inputs_stage_2 * weight2 + inputs_stage_1 * weight1
        #fusion stage2_with_feats3_2
        normed_weight_2 = torch.softmax(torch.stack([self.weight3, self.weight4]), dim=0)
        weight3, weight4 = normed_weight_2[0], normed_weight_2[1]
        inputs_stage_2 = inputs_stage_2*weight3+feats3_2*weight4
        #downsampling
        feats3 = self.conv2_to_3(inputs_stage_2)
        feats3 = self.act2(feats3)
        feats4 = self.conv3_to_4(feats3)

        return feats4, outputs_gcn


    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:

        feats, gcn_2_input = self.gcn_based_multi_scale_feature_fusion(feats)
        feats = self.final_layer(feats)  # -> B, K, H, W
        # flatten the output heatmap
        feats = torch.flatten(feats, 2)
        feats = self.mlp(feats)
        feats = self.gau(feats, gcn_2_input)
        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)

        return pred_x, pred_y
