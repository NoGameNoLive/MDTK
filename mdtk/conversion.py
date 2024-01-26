'''
-*- coding: utf-8 -*-
time: 2023/8/12 11:17
file: convertor.py
author: Endy_Liu_Noonell
'''
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from .bvh_preprocesser import *
from .data import *
from .parsers import *


class BVH2CSMD(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.data_pipe_wpos = Pipeline([
            # 将欧拉角转换为position表示
            ('toPosition', MocapParameterizer_bvh('position')),
            # 过滤末端节点标记点
            ('lerpNub', LerpNub_bvh()),
        ])
        # csmd_path = r'E:\0_Scientific_Research\AGCC\Dataset\csmd_file\test\base\conver_base.csmd'
        csmd_path = r'conver_test\csmd\rest.csmd'
        csmd_parser = CSMDParser()
        self.csmd_base = csmd_parser.parse(csmd_path)
        self.mocap_map = {'Hip': 'Hips',
                          'LeftHip': 'LeftUpLeg',
                          'LeftKnee': 'LeftLeg',
                          'LeftAnkle': 'LeftFoot',
                          'LeftFoot': 'LeftToeBase',
                          'RightHip': 'RightUpLeg',
                          'RightKnee': 'RightLeg',
                          'RightAnkle': 'RightFoot',
                          'RightFoot': 'RightToeBase',
                          'LowSpine': 'LowerBack',
                          'HighSpine': 'Spine',
                          'Chest': 'Spine1',
                          'Neck': 'Neck1',
                          'BackHead': 'Head',
                          'ForeHead': None,
                          'LeftShoulder': 'LeftArm',
                          'LeftElbow': 'LeftForeArm',
                          'LeftWrist': 'LeftHand',
                          'LeftHand': None,
                          'RightShoulder': 'RightArm',
                          'RightElbow': 'RightForeArm',
                          'RightWrist': 'RightHand',
                          'RightHand': None}

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Q = []
        for bvh_container in X:
            csmd_container = self.csmd_base
            pos_container = self.data_pipe_wpos.fit_transform([bvh_container])[0]
            '''reset skeleton'''
            for joint in csmd_container.traverse():
                bvh_joint = self.mocap_map[joint]
                if bvh_joint is not None:
                    bone_vector = np.array(bvh_container.skeleton[bvh_joint]['offsets'])
                    bvh_parent = bvh_container.skeleton[bvh_joint]['parent']
                    if bvh_parent is not None:
                        while bvh_parent != self.mocap_map[csmd_container.skeleton[joint]['parent']]:
                            bone_vector += np.array(bvh_container.skeleton[bvh_parent]['offsets'])
                            bvh_parent = bvh_container.skeleton[bvh_parent]['parent']
                    csmd_container.skeleton[joint]['length'] = np.linalg.norm(bone_vector)

            '''reset basic'''
            csmd_container.time_list = pos_container.values.index.tolist()
            df_index = pos_container.values.index
            df_column = csmd_container.values.columns
            new_values = pd.DataFrame(data=0.0, index=df_index, columns=df_column)
            csmd_container.values = new_values
            root_position = np.array(
                [bvh_container.values['Hips_Xposition'].tolist(), bvh_container.values['Hips_Yposition'].tolist(),
                 bvh_container.values['Hips_Zposition'].tolist()]).T
            csmd_container.root_position = root_position

            lhip_position = np.array(
                [pos_container.values['LeftUpLeg_Xposition'].tolist(),
                 pos_container.values['LeftUpLeg_Yposition'].tolist(),
                 pos_container.values['LeftUpLeg_Zposition'].tolist()]).T
            rhip_position = np.array(
                [pos_container.values['RightUpLeg_Xposition'].tolist(),
                 pos_container.values['RightUpLeg_Yposition'].tolist(),
                 pos_container.values['RightUpLeg_Zposition'].tolist()]).T
            spine_position = np.array([
                pos_container.values['Spine_Xposition'].tolist(),
                pos_container.values['Spine_Yposition'].tolist(),
                pos_container.values['Spine_Zposition'].tolist()]).T
            Xaxis_list = (lhip_position - rhip_position) / np.linalg.norm(lhip_position - rhip_position)
            temp_axis_list = (spine_position - root_position) / np.linalg.norm(spine_position - root_position)

            root_direction = []
            for clip_num in range(len(Xaxis_list)):
                root_direction.append(np.cross(Xaxis_list[clip_num], temp_axis_list[clip_num]) / np.linalg.norm(
                    np.cross(Xaxis_list[clip_num], temp_axis_list[clip_num])))
                # print(root_direction)
            root_direction = np.array(root_direction)
            # csmd_container.root_direction = root_direction

            '''reset motion'''
            # get rotation matrix
            trans_m_list = []
            root_pos_list = []
            dir_m_list = []
            for clip_num in range(len(csmd_container.time_list)):
                realZ = root_direction[clip_num]
                # print(realZ)
                # print(realZ)
                realX = Xaxis_list[clip_num]
                # print(realX)
                realY = np.cross(realZ, realX) / np.linalg.norm(np.cross(realZ, realX))
                # print(realY)
                # print(np.cross(realY, realZ))
                trans_m = [realX, realY, realZ]
                trans_m = np.matrix(trans_m)
                # print(trans_m)
                # clip_time = csmd_container.time_list[clip_num]
                dir_m_list.append(trans_m.T)
                trans_m_list.append(trans_m)
                root_pos_list.append(root_position)
                # print(trans_m)
            # print(trans_m_list)
            csmd_container.root_direction = dir_m_list
            trans_m_array = np.array(trans_m_list)
            # root_pos_array=np.array(root_pos_list)

            for joint in csmd_container.traverse():
                # print(joint)
                bvh_joint = self.mocap_map[joint]
                if bvh_joint == None:
                    csmd_parent = csmd_container.skeleton[joint]['parent']
                    csmd_container.values[joint + '_Xvector'] = csmd_container.values[csmd_parent + '_Xvector']
                    csmd_container.values[joint + '_Yvector'] = csmd_container.values[csmd_parent + '_Yvector']
                    csmd_container.values[joint + '_Zvector'] = csmd_container.values[csmd_parent + '_Zvector']
                    if 'A' in csmd_container.skeleton[joint]['order']:
                        csmd_container.values[joint + '_Arotation'] = csmd_container.values[csmd_parent + '_Arotation']
                    continue
                if joint == csmd_container.root_name:
                    continue
                joint_pos = np.matrix([
                    pos_container.values[bvh_joint + '_Xposition'].tolist(),
                    pos_container.values[bvh_joint + '_Yposition'].tolist(),
                    pos_container.values[bvh_joint + '_Zposition'].tolist()
                ])
                bvh_parent = bvh_container.skeleton[bvh_joint]['parent']
                while bvh_parent != self.mocap_map[csmd_container.skeleton[joint]['parent']]:
                    bvh_parent = bvh_container.skeleton[bvh_parent]['parent']
                parent_pos = np.matrix([
                    pos_container.values[bvh_parent + '_Xposition'].tolist(),
                    pos_container.values[bvh_parent + '_Yposition'].tolist(),
                    pos_container.values[bvh_parent + '_Zposition'].tolist()
                ])
                vector = np.array(joint_pos - parent_pos).T
                # print(vector)
                new_vector = []
                # print(np.matmul(trans_m_array, vector.T))
                if csmd_container.skeleton[joint]['length'] != 0:
                    vector = vector / csmd_container.skeleton[joint]['length']
                for i in range(len(vector)):
                    temp_vector = np.matmul(trans_m_array[i], vector[i])
                    new_vector.append(temp_vector)
                new_vector = np.array(new_vector).T
                csmd_container.values[joint + '_Xvector'] = new_vector[0]
                csmd_container.values[joint + '_Yvector'] = new_vector[1]
                csmd_container.values[joint + '_Zvector'] = new_vector[2]
                if 'A' in csmd_container.skeleton[joint]['order']:
                    child = bvh_container.skeleton[bvh_joint]['children'][0]
                    bone_axis = bvh_container.skeleton[child]['offsets']
                    if np.linalg.norm(bone_axis) == 0:
                        bone_axis = bvh_container.skeleton[bvh_joint]['offsets']
                    bone_axis = bone_axis / np.linalg.norm(bone_axis)
                    axis_map = {0: 'Z', 1: 'Y', 2: 'X'}
                    rotate_axis = None
                    for item in range(len(bone_axis)):
                        if math.fabs(bone_axis[item]) >= 0.90:
                            rotate_axis = axis_map[item]
                    if rotate_axis is not None:
                        csmd_container.values[joint + '_Arotation'] = \
                            bvh_container.values[
                                bvh_joint + '_' + rotate_axis + 'rotation']
            Q.append(csmd_container)
        return Q
