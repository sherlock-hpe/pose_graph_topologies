import torch
import torch.nn.functional as F
from networkx.classes import edges
from torch import dtype
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from mmpose.evaluation import keypoint_epe
from mmpose.registry import MODELS
from mmpose.models.utils.rtmcc_block import ScaleNorm
from typing import Tuple

@MODELS.register_module()
#定义图卷积
class GCNLayerTwo(torch.nn.Module):
    def __init__(self, dataset_type):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_type = dataset_type
        self.conv1 = GCNConv(256, 256)
        self.conv2 = GCNConv(256, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.act1 = torch.nn.ReLU(inplace=True)


    def forward(self, feats):
        x = feats
        batch_size, c, _ = x.shape
        data_key = build_keypoints_adj_matrix(x, self.data_type, self.device)
        x, edge_index_key = data_key.x, data_key.edge_index

        x = self.conv1(x, edge_index_key)
        x = self.bn1(x)
        x = self.act1(x)

        x = x.view(batch_size, c, -1).contiguous()

        data_key = bulid_keypoints_adj_matrix_reverse(x, self.data_type, self.device)
        x, edge_index_key = data_key.x, data_key.edge_index

        x = self.conv2(x, edge_index_key)

        x = x.view(batch_size, c, -1).contiguous()

        return x

