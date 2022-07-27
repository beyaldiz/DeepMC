'''
This script contains the model components of MoCap_Solver.
This script is borrowed from [Chen et al. 2021].
'''
import torch
from torch.nn import Linear, BatchNorm1d, LeakyReLU, Flatten
import torch.nn as nn
import numpy as np


from deepmc.models.components.MoCap_Solver.skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, SkeletonLinear, build_edge_topology


class TS_enc(nn.Module):
    def __init__(
        self,
        topology: list = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
        num_layers: int = 2,
        skeleton_dist: int = 2,
    ):
        super(TS_enc, self).__init__()
        edges = build_edge_topology(topology, torch.zeros((len(topology), 3)))
        self.channel_list = []
        self.topologies = [edges]
        self.edge_num = [len(edges) + 1]
        self.layers = nn.ModuleList()
        self.pooling_list = []
        activation = nn.LeakyReLU(negative_slope=0.2)
        channels = 3

        for i in range(num_layers):
            neighbor_list = find_neighbor(edges, skeleton_dist)
            seq = []
            if i == 0: self.channel_list.append(channels * len(neighbor_list))

            seq.append(SkeletonLinear(neighbor_list, in_channels=channels * len(neighbor_list),
                                      out_channels=channels * 2 * len(neighbor_list), extra_dim1=True))
            if i < num_layers - 1:
                pool = SkeletonPool(edges, channels_per_edge=channels * 2, pooling_mode='mean')
                seq.append(pool)
                edges = pool.new_edges
                self.pooling_list.append(pool.pooling_list)
                self.topologies.append(pool.new_edges)
                self.edge_num.append(len(self.topologies[-1]) + 1)
            seq.append(activation)
            self.channel_list.append(channels * 2 * len(neighbor_list))
            channels *= 2
            self.layers.append(nn.Sequential(*seq))

    # input should have shape B * E * 3
    def forward(self, input):
        output = [input]
        for i, layer in enumerate(self.layers):
            input = layer(input)
            output.append(input.squeeze())
        return output


class TS_dec(nn.Module):
    def __init__(
        self,
        topology: list = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
        num_layers: int = 2,
        skeleton_dist: int = 2,
    ):
        super(TS_dec, self).__init__()
        enc = TS_enc(topology, num_layers, skeleton_dist)
        self.layers = nn.ModuleList()
        activation = nn.LeakyReLU(negative_slope=0.2)
        channels = 3
        self.channel_list = []
        self.mclayers = nn.ModuleList()
        self.mcnormLayers = nn.ModuleList()
        self.mcactLayers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = enc.channel_list[num_layers - i]
            out_channels = in_channels // 2
            neighbor_list = find_neighbor(enc.topologies[num_layers - i - 1], skeleton_dist)
            seq = []

            if i > 0:
                unpool = SkeletonUnpool(enc.pooling_list[num_layers - i - 1], in_channels // len(neighbor_list))

                seq.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=False))
                in_channels = in_channels * 2
                seq.append(unpool)
                out_channels = enc.channel_list[num_layers - i - 1]

            seq.append(SkeletonLinear(neighbor_list, in_channels=in_channels,
                                      out_channels=out_channels, extra_dim1=True))
            # if i != args.num_layers - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))

    # input should have shape B * E * 3
    def forward(self, input):
        output = [input]
        for i, layer in enumerate(self.layers):
            input = layer(input)
            output.append(input.squeeze())
        return output


class MC_enc(nn.Module):
    def __init__(
        self,
        num_markers: int = 56,
        num_joints: int = 24
    ):
        super(MC_enc, self).__init__()
        self.layers = nn.ModuleList()
        self.normLayers = nn.ModuleList()
        self.actLayers = nn.ModuleList()
        self.prevLayer0 = Flatten()
        self.prevLayer = nn.Sequential(
            Linear(num_markers * num_joints * 3 + num_joints * 3, 1024))
        self.prevLayer1 = nn.Sequential(
            Linear(1024 + 168, 1024))
        self.layers.append(Linear(1024, 1024))
        self.normLayers.append(BatchNorm1d(1024))
        self.actLayers.append(LeakyReLU(0.5))
        self.layers.append(Linear(1024, 1024))
        self.normLayers.append(BatchNorm1d(1024))
        self.actLayers.append(LeakyReLU(0.5))


    def forward(self, input, TS_input, TS_latent):
        input3 = self.prevLayer0(TS_latent)
        input0 = self.prevLayer0(input)
        input1 = self.prevLayer0(TS_input)
        input2 = torch.cat([input0, input1], dim=1)
        input2 = self.prevLayer(input2)
        x0 = self.normLayers[0](input2)
        x0 = self.actLayers[0](x0)
        x = self.layers[0](x0)
        m0 = x + x0

        m0 = torch.cat([m0, input3], dim=1)
        m0 = self.prevLayer1(m0)

        x1 = self.normLayers[1](m0)
        x1 = self.actLayers[1](x1)
        x = self.layers[1](x1)
        m1 = x + x1

        return m1


class MC_dec(nn.Module):
    def __init__(
        self,
        num_markers: int = 56,
        num_joints: int = 24
    ):
        super(MC_dec, self).__init__()
        # self.enc = mc_enc
        self.num_markers = num_markers
        self.num_joints = num_joints
        self.configlayers = nn.ModuleList()
        self.confignormLayers = nn.ModuleList()
        self.configactLayers = nn.ModuleList()
        self.prevLayer0 = Flatten()
        self.prevLayer1 = nn.Sequential(
            Linear(1024 + 168, 1024))
        self.prevLayer2 = nn.Sequential(
            Linear(1024 + 24 * 3, 1024))

        self.configlayers.append(Linear(1024, 1024))
        self.confignormLayers.append(BatchNorm1d(1024))
        self.configactLayers.append(LeakyReLU(0.5))

        self.configlayers.append(Linear(1024, 1024))
        self.confignormLayers.append(BatchNorm1d(1024))
        self.configactLayers.append(LeakyReLU(0.5))

        self.configactLayers.append(LeakyReLU(0.5))
        self.configlayers.append(Linear(1024, num_markers * num_joints * 3))

        # self.configSoftmax = nn.Softmax(dim=2)

    def forward(self, input, TS_input, TS_latent):
        TS_input = self.prevLayer0(TS_input)
        TS_latent = self.prevLayer0(TS_latent)
        input0 = torch.cat([input, TS_latent], dim=1)
        input1 = self.prevLayer1(input0)
        x_vector = self.confignormLayers[0](input1)
        x_vector = self.configactLayers[0](x_vector)
        x_vector1 = self.configlayers[0](x_vector)
        x_vector = x_vector + x_vector1

        x_vector = torch.cat([x_vector, TS_input], dim=1)
        x_vector = self.prevLayer2(x_vector)

        x_vector = self.confignormLayers[1](x_vector)
        x_vector = self.configactLayers[1](x_vector)
        x_vector1 = self.configlayers[1](x_vector)
        x_vector = x_vector + x_vector1

        x_vector = self.configactLayers[2](x_vector)
        mrk_config_x_o = self.configlayers[2](x_vector)
        mrk_config_x_o = mrk_config_x_o.reshape([mrk_config_x_o.shape[0], self.num_markers, self.num_joints, 3])
        return mrk_config_x_o


class MO_enc(nn.Module):
    def __init__(
        self,
        topology: list = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
        num_layers: int = 2,
        kernel_size: int = 15,
        skeleton_dist: int = 2,
        extra_conv: int = 0,
    ):
        super(MO_Enc, self).__init__()
        self.topologies = [topology]
        self.channel_base = [4]
        self.channel_list = []
        self.edge_num = [len(topology) + 1]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.convs = []

        padding = (kernel_size - 1) // 2
        bias = True
        add_offset = True

        for i in range(num_layers):
            self.channel_base.append(self.channel_base[-1] * 2)

        for i in range(num_layers):
            seq = []
            neighbor_list = find_neighbor(self.topologies[i], skeleton_dist)
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i+1] * self.edge_num[i]
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            for _ in range(extra_conv):
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=self.edge_num[i], kernel_size=kernel_size, stride=1,
                                        padding=padding, padding_mode='reflection', bias=bias))
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.edge_num[i], kernel_size=kernel_size, stride=2,
                                    padding=padding, padding_mode='reflection', bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))
            self.convs.append(seq[-1])
            last_pool = True if i == num_layers - 1 else False
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode='mean',
                                channels_per_edge=out_channels // len(neighbor_list), last_pool=last_pool)
            seq.append(pool)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]) + 1)
            if i == num_layers - 1:
                self.last_channel = self.edge_num[-1] * self.channel_base[i + 1]

    def forward(self, input, offset=None):
        # padding the one zero row to global position, so each joint including global position has 4 channels as input
        input = torch.cat((input, torch.zeros_like(input[:, [0], :])), dim=1)

        for i, layer in enumerate(self.layers):
            if offset is not None:
                self.convs[i].set_offset(offset[i])
            input = layer(input)
        return input


class MO_Dec(nn.Module):
    def __init__(
        self,
        topology: list = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
        num_layers: int = 2,
        kernel_size: int = 15,
        skeleton_dist: int = 2,
        extra_conv: int = 0,
    ):
        super(MO_Dec, self).__init__()
        self.layers = nn.ModuleList()
        self.unpools = nn.ModuleList()
        enc = MO_Enc(topology=topology, num_layers=num_layers, kernel_size=kernel_size,
                        skeleton_dist=skeleton_dist,extra_conv=extra_conv)
        self.convs = []

        padding = (kernel_size - 1) // 2

        add_offset = True

        for i in range(num_layers):
            seq = []
            in_channels = enc.channel_list[num_layers - i]
            out_channels = in_channels // 2
            neighbor_list = find_neighbor(enc.topologies[num_layers - i - 1], skeleton_dist)

            if i != 0 and i != num_layers - 1:
                bias = False
            else:
                bias = True

            self.unpools.append(SkeletonUnpool(enc.pooling_list[num_layers - i - 1], in_channels // len(neighbor_list)))

            seq.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=False))
            seq.append(self.unpools[-1])
            for _ in range(extra_conv):
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=enc.edge_num[num_layers - i - 1], kernel_size=kernel_size,
                                        stride=1,
                                        padding=padding, padding_mode='reflection', bias=bias))
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=enc.edge_num[num_layers - i - 1], kernel_size=kernel_size, stride=1,
                                    padding=padding, padding_mode='reflection', bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * enc.channel_base[num_layers - i - 1] // enc.channel_base[0]))
            self.convs.append(seq[-1])
            if i != num_layers - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))

            self.layers.append(nn.Sequential(*seq))

    def forward(self, input, offset=None):
        for i, layer in enumerate(self.layers):
            self.convs[i].set_offset(offset[len(self.layers) - i - 1])
            input = layer(input)
        # throw the padded rwo for global position
        input = input[:, :-1, :]

        return input