import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from deepmc.models.components.vnn import *


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None:  # dynamic knn graph
            idx = knn(x, k=k)
        else:  # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature


class VN_DGCNN_pose(nn.Module):
    def __init__(self, n_knn: int = 20, pooling: str = "mean"):
        super(VN_DGCNN_pose, self).__init__()
        self.n_knn = n_knn

        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 128 // 3)
        self.conv4 = VNLinearLeakyReLU(128 // 3 * 2, 256 // 3)

        self.conv5 = VNLinearLeakyReLU(
            256 // 3 + 128 // 3 + 64 // 3 * 2, 1024 // 3, dim=4, share_nonlinearity=True
        )

        self.max_pool = VNMaxPool(2048 // 3)

        self.linear1 = VNLinearLeakyReLU(2048 // 3, 512, dim=3)
        self.linear2 = VNLinearLeakyReLU(512, 256, dim=3)
        self.linear3 = VNLinear(256, 3)

        if pooling == "max":
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(128 // 3)
            self.pool4 = VNMaxPool(256 // 3)
        elif pooling == "mean":
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv2(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv3(x)
        x3 = self.pool3(x)

        x = get_graph_feature(x3, k=self.n_knn)
        x = self.conv4(x)
        x4 = self.pool4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)


        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)

        # x1 = self.max_pool(x)
        x2 = mean_pool(x)
        # x = torch.cat((x1, x2), 1)

        x = x2
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

        return x.transpose(-1, -2)


class VN_DGCNN_pose_seg(nn.Module):
    def __init__(self, n_knn: int = 20, pooling: str = "mean"):
        super(VN_DGCNN_pose_seg, self).__init__()
        self.n_knn = n_knn
        
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)
        
        self.conv1 = VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv4 = VNLinearLeakyReLU(64//3, 64//3)
        self.conv5 = VNLinearLeakyReLU(64//3*2, 64//3)
        
        if pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(64//3)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
        
        self.conv6 = VNLinearLeakyReLU(64//3*3, 1024//3, dim=4, share_nonlinearity=True)
        self.linear_temp = VNLinear(2048 // 3, 3)
        # self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False)
        # self.conv8 = nn.Sequential(nn.Conv1d(2299, 256, kernel_size=1, bias=False),
        #                        self.bn8,
        #                        nn.LeakyReLU(negative_slope=0.2))
        
        # self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
        #                            self.bn7,
        #                            nn.LeakyReLU(negative_slope=0.2))
        
        # self.dp1 = nn.Dropout(p=0.5)
        # self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
        #                            self.bn9,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.dp2 = nn.Dropout(p=0.5)
        # self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
        #                            self.bn10,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv11 = nn.Conv1d(128, num_part, kernel_size=1, bias=False)
        

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        
        x = x.unsqueeze(1)
        
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        
        x123 = torch.cat((x1, x2, x3), dim=1)
        
        x = self.conv6(x123)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1).mean(-1)

        x = self.linear_temp(x)

        return x.transpose(-1, -2)
        # x, z0 = self.std_feature(x)
        # x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_points)
        # x = x.view(batch_size, -1, num_points)
        # x = x.max(dim=-1, keepdim=True)[0]

        # l = l.view(batch_size, -1, 1)
        # l = self.conv7(l)

        # x = torch.cat((x, l), dim=1)
        # x = x.repeat(1, 1, num_points)

        # x = torch.cat((x, x123), dim=1)

        # x = self.conv8(x)
        # x = self.dp1(x)
        # x = self.conv9(x)
        # x = self.dp2(x)
        # x = self.conv10(x)
        # x = self.conv11(x)
        
        # trans_feat = None
        return x.transpose(1, 2), trans_feat

