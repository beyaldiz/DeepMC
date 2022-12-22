'''
This script contains the model components of MoCap_Solver.
This script is borrowed from [Chen et al. 2021].
'''
import torch
from torch.nn import Linear, BatchNorm1d, LeakyReLU, Flatten
import torch.nn as nn
import numpy as np


from deepmc.models.components.MoCap_Solver.skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, SkeletonLinear, build_edge_topology
from deepmc.models.components.vnn import VNLinear, VNLinearAndLeakyReLU, VNBatchNorm, VNLeakyReLU, VNStdFeature
from deepmc.utils.MoCap_Solver.utils import _dict_marker, _dict_joints


class VN_Marker_enc(nn.Module):
    def __init__(
        self,
        num_markers: int = 56,
        num_joints: int = 24,
        num_feat: int = 2048
    ):
        super(VN_Marker_enc, self).__init__()
        # Hard coded
        FRAMENUM = 64
        self.gamma_para = 0.001
        self.HUBER_DELTA = 200
        self.lambda_para = np.ones((num_markers, 3))
        self.lambda_jt_para = np.ones((num_joints, 3))
        self.head_idx = [_dict_marker["ARIEL"], _dict_marker["LBHD"], _dict_marker["LFHD"], _dict_marker["RFHD"],
                    _dict_marker["RBHD"]]
        self.head_jt_idx = [_dict_joints["Head"]]
        self.shoulder_idx = [_dict_marker["LTSH"], _dict_marker["LBSH"], _dict_marker["LFSH"], _dict_marker["RTSH"],
                        _dict_marker["RFSH"], _dict_marker["RBSH"]]
        self.shoulder_jt_idx = [_dict_joints['L_Collar'], _dict_joints['R_Collar'], _dict_joints['L_Shoulder'], _dict_joints['R_Shoulder']]
        self.arm_idx = [
            _dict_marker["LUPA"], _dict_marker["LIEL"], _dict_marker["LELB"], _dict_marker["LWRE"],
            _dict_marker["RUPA"], _dict_marker["RIEL"], _dict_marker["RELB"], _dict_marker["RWRE"]
        ]
        self.arm_jt_idx = [_dict_joints['L_Elbow'], _dict_joints['R_Elbow']]
        self.wrist_hand_idx = [
            _dict_marker["LOWR"], _dict_marker["LIWR"], _dict_marker["LIHAND"], _dict_marker["LOHAND"],
            _dict_marker["ROWR"], _dict_marker["RIWR"], _dict_marker["RIHAND"], _dict_marker["ROHAND"]
        ]
        self.wrist_hand_jt_idx = [_dict_joints['L_Wrist'], _dict_joints['R_Wrist'], _dict_joints['L_Hand'], _dict_joints['R_Hand']]

        self.torso_idx = [
            _dict_marker["CLAV"], _dict_marker["STRN"], _dict_marker["C7"], _dict_marker["T10"], _dict_marker["L4"],
            _dict_marker["LMWT"], _dict_marker["LFWT"], _dict_marker["LBWT"],
            _dict_marker["RMWT"], _dict_marker["RFWT"], _dict_marker["RBWT"]
        ]

        self.torso_jt_idx = [_dict_joints['Pelvis'],_dict_joints['Spine1'], _dict_joints['Spine2'], _dict_joints['Spine3'], _dict_joints['L_Hip'], _dict_joints['R_Hip'], _dict_joints['Neck']]
        self.thigh_idx = [
            _dict_marker["LKNI"], _dict_marker["LKNE"], _dict_marker["LHIP"], _dict_marker["LSHN"],
            _dict_marker["RKNI"], _dict_marker["RKNE"], _dict_marker["RHIP"], _dict_marker["RSHN"]
        ]
        self.thigh_jt_idx = [_dict_joints['L_Knee'], _dict_joints['R_Knee']]

        self.foots_idx = [
            _dict_marker["LANK"], _dict_marker["LHEL"], _dict_marker["LMT1"], _dict_marker["LTOE"],
            _dict_marker["LMT5"],
            _dict_marker["RANK"], _dict_marker["RHEL"], _dict_marker["RMT1"], _dict_marker["RTOE"], _dict_marker["RMT5"]
        ]
        self.foots_jt_idx = [_dict_joints['L_Ankle'], _dict_joints['R_Ankle'], _dict_joints['L_Foot'], _dict_joints['R_Foot']]

        self.lambda_para[self.head_idx] = self.lambda_para[self.head_idx] * 10  # head
        self.lambda_para[self.shoulder_idx] = self.lambda_para[self.shoulder_idx] * 5  # shoulder
        self.lambda_para[self.arm_idx] = self.lambda_para[self.arm_idx] * 8  # arm
        self.lambda_para[self.wrist_hand_idx] = self.lambda_para[self.wrist_hand_idx] * 10  # wrist
        self.lambda_para[self.torso_idx] = self.lambda_para[self.torso_idx] * 5  # torso
        self.lambda_para[self.thigh_idx] = self.lambda_para[self.thigh_idx] * 8  # thigh
        self.lambda_para[self.foots_idx] = self.lambda_para[self.foots_idx] * 10  # foots

        self.lambda_jt_para[self.head_jt_idx] = self.lambda_jt_para[self.head_jt_idx] * 10  # head
        self.lambda_jt_para[self.shoulder_jt_idx] = self.lambda_jt_para[self.shoulder_jt_idx] * 5  # shoulder
        self.lambda_jt_para[self.arm_jt_idx] = self.lambda_jt_para[self.arm_jt_idx] * 8  # arm
        self.lambda_jt_para[self.wrist_hand_jt_idx] = self.lambda_jt_para[self.wrist_hand_jt_idx] * 10  # wrist
        self.lambda_jt_para[self.torso_jt_idx] = self.lambda_jt_para[self.torso_jt_idx] * 5  # torso
        self.lambda_jt_para[self.thigh_jt_idx] = self.lambda_jt_para[self.thigh_jt_idx] * 8  # thigh
        self.lambda_jt_para[self.foots_jt_idx] = self.lambda_jt_para[self.foots_jt_idx] * 10  # foots
        self.joints_nums = num_joints
        self.marker_nums = num_markers
        self.layers = nn.ModuleList()
        self.normLayers = nn.ModuleList()
        self.actLayers = nn.ModuleList()
        self.prevLayer = VNLinear(FRAMENUM * num_markers, num_feat)
        self.layers.append(VNLinear(num_feat, num_feat))
        self.normLayers.append(VNBatchNorm(num_feat, dim=3))
        self.actLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))
        self.layers.append(VNLinear(num_feat, num_feat))
        self.normLayers.append(VNBatchNorm(num_feat, dim=3))
        self.actLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))
        self.layers.append(VNLinear(num_feat, num_feat))
        self.normLayers.append(VNBatchNorm(num_feat, dim=3))
        self.actLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))

        self.actLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))
        self.layers.append(VNLinear(num_feat, num_feat))
        self.actLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))

    def forward(self, input):
        input = input.view(input.shape[0], -1, input.shape[3])
        input1 = self.prevLayer(input)
        x0 = self.normLayers[0](input1)
        x0 = self.actLayers[0](x0)
        x = self.layers[0](x0)
        m0 = x + x0
        x1 = self.normLayers[1](m0)
        x1 = self.actLayers[1](x1)
        x = self.layers[1](x1)
        m1 = x + x1
        x2 = self.normLayers[2](m1)
        x2 = self.actLayers[2](x2)
        x = self.layers[2](x2)
        m2 = x + x2
        x3 = self.normLayers[2](m2)
        x3 = self.actLayers[3](x3)
        x3 = self.layers[3](x3)
        x = self.actLayers[4](x3)
        return x


class VN_Marker_dec(nn.Module):
    def __init__(
        self,
        num_feat: int = 2048
    ):
        super(VN_Marker_dec, self).__init__()
        self.configlayers = nn.ModuleList()
        self.confignormLayers = nn.ModuleList()
        self.configactLayers = nn.ModuleList()
        self.mdlayers = nn.ModuleList()
        self.mdnormLayers = nn.ModuleList()
        self.mdactLayers = nn.ModuleList()
        self.offsetlayers = nn.ModuleList()
        self.offsetnormLayers = nn.ModuleList()
        self.offsetactLayers = nn.ModuleList()
        self.transLayers = nn.ModuleList()
        self.transactLayers = nn.ModuleList()

        self.configlayers.append(VNLinear(num_feat, num_feat))
        self.confignormLayers.append(VNBatchNorm(num_feat, dim=3))
        self.configactLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))

        self.configlayers.append(VNLinear(num_feat, num_feat))
        self.confignormLayers.append(VNBatchNorm(num_feat, dim=3))
        self.configactLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))

        self.configactLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))
        self.configlayers.append(VNLinear(num_feat, 1024))

        self.mdlayers.append(VNLinear(num_feat, num_feat))
        self.mdnormLayers.append(VNBatchNorm(num_feat, dim=3))
        self.mdactLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))

        self.mdlayers.append(VNLinear(num_feat, num_feat))
        self.mdnormLayers.append(VNBatchNorm(num_feat, dim=3))
        self.mdactLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))



        self.mdlayers.append(VNLinear(num_feat, num_feat))
        self.mdnormLayers.append(VNBatchNorm(num_feat, dim=3))
        self.mdactLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))

        self.mdactLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))
        self.mdlayers.append(VNLinear(num_feat, 2048))


        self.offsetlayers.append(VNLinear(num_feat, num_feat))
        self.offsetnormLayers.append(VNBatchNorm(num_feat, dim=3))
        self.offsetactLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))

        self.offsetlayers.append(VNLinear(num_feat, num_feat))
        self.offsetnormLayers.append(VNBatchNorm(num_feat, dim=3))
        self.offsetactLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))

        self.offsetlayers.append(VNLinear(num_feat, num_feat))
        self.offsetnormLayers.append(VNBatchNorm(num_feat, dim=3))
        self.offsetactLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))

        self.offsetactLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))
        self.offsetlayers.append(VNLinear(num_feat, 168))




        self.inv1 = VNStdFeature(1024, dim=3)
        self.inv2 = VNStdFeature(2048, dim=3)
        self.inv3 = VNStdFeature(168, dim=3)

        self.transform1 = Linear(1024*3, 1024)
        self.transform2 = Linear(2048*3, 2048)
        self.transform3 = Linear(168*3, 168)


    def forward(self, input):
        ######marker config#######################
        x_vector = self.confignormLayers[0](input)
        x_vector = self.configactLayers[0](x_vector)
        x_vector1 = self.configlayers[0](x_vector)
        x_vector = x_vector + x_vector1

        x_vector = self.confignormLayers[1](x_vector)
        x_vector = self.configactLayers[1](x_vector)
        x_vector1 = self.configlayers[1](x_vector)
        x_vector = x_vector + x_vector1

        x_vector = self.configactLayers[2](x_vector)
        mrk_config_x_o = self.configlayers[2](x_vector)

        # Shape: BS x 341 x 3
        mrk_config_x_o, _ = self.inv1(mrk_config_x_o)
        mrk_config_x_o = self.transform1(mrk_config_x_o.view(mrk_config_x_o.shape[0], -1))
        
        mrk_config_x_o = mrk_config_x_o.reshape([mrk_config_x_o.shape[0], 1024])

        ######motion data#############################
        md_vector = self.mdnormLayers[0](input)
        md_vector = self.mdactLayers[0](md_vector)
        md_vector1 = self.mdlayers[0](md_vector)
        md_vector = md_vector + md_vector1

        md_vector = self.mdnormLayers[1](md_vector)
        md_vector = self.mdactLayers[1](md_vector)
        md_vector1 = self.mdlayers[1](md_vector)
        md_vector = md_vector + md_vector1

        md_vector = self.mdnormLayers[2](md_vector)
        md_vector = self.mdactLayers[2](md_vector)
        md_vector1 = self.mdlayers[2](md_vector)
        md_vector = md_vector + md_vector1

        md_vector5 = self.mdactLayers[3](md_vector)
        md_x_o = self.mdlayers[3](md_vector5)

        # Shape BS x 682 x 3
        md_x_o, _ = self.inv2(md_x_o)
        md_x_o = self.transform2(md_x_o.view(md_x_o.shape[0], -1))

        md_x_o = md_x_o.reshape([md_x_o.shape[0], 2048])


        ######offset data#############################
        offset_vector = self.offsetnormLayers[0](input)
        offset_vector = self.offsetactLayers[0](offset_vector)
        offset_vector1 = self.offsetlayers[0](offset_vector)
        offset_vector = offset_vector + offset_vector1

        offset_vector = self.offsetnormLayers[1](offset_vector)
        offset_vector = self.offsetactLayers[1](offset_vector)
        offset_vector1 = self.offsetlayers[1](offset_vector)
        offset_vector = offset_vector + offset_vector1

        offset_vector = self.offsetnormLayers[2](offset_vector)
        offset_vector = self.offsetactLayers[2](offset_vector)
        offset_vector1 = self.offsetlayers[2](offset_vector)
        offset_vector = offset_vector + offset_vector1

        offset_vector = self.offsetactLayers[3](offset_vector)
        offset_x_o = self.offsetlayers[3](offset_vector)

        offset_x_o, _ = self.inv3(offset_x_o)
        offset_x_o = self.transform3(offset_x_o.view(offset_x_o.shape[0], -1))
        
        offset_x_o = offset_x_o.reshape([offset_x_o.shape[0], 168])

        return mrk_config_x_o, md_x_o, offset_x_o

class VN_Marker_dec_root(nn.Module):
    def __init__(
        self,
        num_feat: int = 2048
    ):
        super(VN_Marker_dec_root, self).__init__()
        self.mdlayers = nn.ModuleList()
        self.mdnormLayers = nn.ModuleList()
        self.mdactLayers = nn.ModuleList()

        self.mdlayers.append(VNLinear(num_feat, num_feat))
        self.mdnormLayers.append(VNBatchNorm(num_feat, dim=3))
        self.mdactLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))

        self.mdlayers.append(VNLinear(num_feat, num_feat))
        self.mdnormLayers.append(VNBatchNorm(num_feat, dim=3))
        self.mdactLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))



        self.mdlayers.append(VNLinear(num_feat, num_feat))
        self.mdnormLayers.append(VNBatchNorm(num_feat, dim=3))
        self.mdactLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))

        self.mdactLayers.append(VNLeakyReLU(num_feat, negative_slope=0.5))
        self.mdlayers.append(VNLinear(num_feat, 256))


    def forward(self, input):
        ######motion data#############################
        md_vector = self.mdnormLayers[0](input)
        md_vector = self.mdactLayers[0](md_vector)
        md_vector1 = self.mdlayers[0](md_vector)
        md_vector = md_vector + md_vector1

        md_vector = self.mdnormLayers[1](md_vector)
        md_vector = self.mdactLayers[1](md_vector)
        md_vector1 = self.mdlayers[1](md_vector)
        md_vector = md_vector + md_vector1

        md_vector = self.mdnormLayers[2](md_vector)
        md_vector = self.mdactLayers[2](md_vector)
        md_vector1 = self.mdlayers[2](md_vector)
        md_vector = md_vector + md_vector1

        md_vector5 = self.mdactLayers[3](md_vector)
        md_x_o = self.mdlayers[3](md_vector5)

        # Shape BS x 256 x 3
        R, t = md_x_o[:, 64:], md_x_o[:, :64]
        R = R.view(-1, 64, 3, 3)
        R = R.transpose(-1, -2).clone()

        return R, t