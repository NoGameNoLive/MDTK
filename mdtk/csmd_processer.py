'''
-*- coding: utf-8 -*-
time: 2023/11/10 17:16
file: csmd_processer.py
author: Endy_Liu_Noonell
'''

import copy
import pandas as pd
import numpy as np
import transforms3d as t3d
import scipy.ndimage.filters as filters
import math

from sklearn.base import BaseEstimator, TransformerMixin

from .rotation_tools import Rotation, euler2expmap, euler2expmap2, expmap2euler, euler_reorder, unroll
from .Quaternions import Quaternions
from .Pivots import Pivots
from .parsers import BVHParser
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图


# pd.set_option('display.max_columns', None)  # 显示完整的列


# pd.set_option('display.max_rows', None)  # 显示完整的行
# pd.set_option('display.expand_frame_repr', False)  # 设置不折叠数据


class MocapParameterizer_csmd(BaseEstimator, TransformerMixin):
    def __init__(self, param_type='euler'):
        '''

        param_type = {'euler', 'quat', 'expmap', 'position', 'expmap2pos'}
        '''
        self.param_type = param_type

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # print("MocapParameterizer: " + self.param_type)
        if self.param_type == 'euler':
            # not ready
            return X
        elif self.param_type == 'expmap':
            # not ready
            return self._to_expmap(X)
        elif self.param_type == 'quat':
            # not ready
            return X
        elif self.param_type == 'position':
            return self._to_pos(X)
        else:
            raise 'param types: euler, quat, expmap, position'

    #        return X

    def inverse_transform(self, X, copy=None):
        print('InverseProcess')
        if self.param_type == 'euler':
            return X
        elif self.param_type == 'expmap':
            return X
        elif self.param_type == 'quat':
            raise 'quat2euler is not supported'
        elif self.param_type == 'position':
            # print('pos_to_euler')
            return self._pos_to_euler(X)
        else:
            raise 'param types: euler, quat, expmap, position'

    def _to_pos(self, X):
        '''Converts joints rotations in Euler angles to joint positions'''

        Q = []
        for track in X:
            pos_channels = []
            root_position = np.array(track.root_position).T
            for channel in track.values.columns:
                if 'Arotation' in channel:
                    continue
                pos_channels.append(channel.replace('vector', 'position'))
            pos_df = pd.DataFrame(index=track.values.index, columns=pos_channels)
            # finish_df = pd.DataFrame(index=track.values.index, columns=pos_channels)

            # Zaxis_list = np.array(
            #     [track.values['LeftUpLeg_Xposition'].tolist() - pos_df.values['RightUpLeg_Xposition'].tolist(),
            #      pos_df.values['LeftUpLeg_Yposition'].tolist() - pos_df.values['RightUpLeg_Yposition'].tolist(),
            #      pos_df.values['LeftUpLeg_Zposition'].tolist() - pos_df.values['RightUpLeg_Zposition'].tolist()]).T
            #
            # for clip_num in range(len(csmd_container.time_list)):
            #     realX = track.root_direction[clip_num]
            #     realZ = Zaxis_list[clip_num]
            #     realY = np.cross(realX, realZ) / np.linalg.norm(np.cross(realX, realZ))
            #     trans_m = [realX, realY, realZ]
            #     trans_m = np.matrix(trans_m).I
            #     clip_time = csmd_container.time_list[clip_num]
            #
            # realX = track.root_direction
            #
            # realY = np.cross(realX, realZ) / np.linalg.norm(np.cross(realX, realZ))
            # trans_m = [realX, realY, realZ]

            for joint in track.traverse():
                new_track = track.clone()
                if joint == track.root_name:
                    continue
                parent = track.skeleton[joint]['parent']
                if parent == track.root_name:
                    pos_df[joint + '_Xposition'] = np.array(track.values[joint + '_Xvector'].tolist()) + \
                                                   root_position[0]
                    pos_df[joint + '_Yposition'] = np.array(track.values[joint + '_Yvector'].tolist()) + \
                                                   root_position[1]
                    pos_df[joint + '_Zposition'] = np.array(track.values[joint + '_Zvector'].tolist()) + \
                                                   root_position[2]
                else:
                    pos_df[joint + '_Xposition'] += np.array(track.values[joint + '_Xvector'].tolist()) + \
                                                    pos_df[parent + '_Xposition']
                    pos_df[joint + '_Yposition'] += np.array(track.values[joint + '_Xvector'].tolist()) + \
                                                    pos_df[parent + '_Yposition']
                    pos_df[joint + '_Zposition'] += np.array(track.values[joint + '_Zvector'].tolist()) + \
                                                    pos_df[parent + '_Zposition']
                pos_rela = np.array([pos_df[joint + '_Xposition'], pos_df[joint + '_Yposition'],
                                     pos_df[joint + '_Zposition']]).T
                pos_world=[]
                for i in range(len(pos_rela)):
                    temp_vector = np.matmul(track.root_direction[i], pos_rela[i])
                    pos_world.append(temp_vector)
                pos_world=np.array(pos_world).T
                pos_df[joint + '_Xposition'] = pos_world[0]
                pos_df[joint + '_Yposition'] = pos_world[1]
                pos_df[joint + '_Zposition'] = pos_world[2]
                # finish_df[joint + '_Xposition'] = pos_world[0]
                # finish_df[joint + '_Yposition'] = pos_world[1]
                # finish_df[joint + '_Zposition'] = pos_world[2]
            new_track.values = pos_df
            Q.append(new_track)
        return Q

    def _pos_to_euler(self, X):
        Q = []
        for track in X:
            new_track = track.clone()
            # print(track.skeleton)
            new_track.values = euler_df
            Q.append(new_track)
        return Q


class Mirror_csmd(BaseEstimator, TransformerMixin):
    def __init__(self, axis="X", append=True):
        """
        Mirrors the data
        """
        self.axis = axis
        self.append = append

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Mirror: " + self.axis)
        Q = []

        if self.append:
            for track in X:
                Q.append(track)

        for track in X:
            channels = []
            titles = []

            new_df = track.clone()
            lft_joints = (joint for joint in track.skeleton if 'Left' in joint)

            new_track = track.clone()

            for lft_joint in lft_joints:
                # lr = euler_df[[c for c in rots if lft_joint + "_" in c]]
                # rot_order = track.skeleton[lft_joint]['order']
                # lft_eulers = [[f[1]['%s_Xrotation'%lft_joint], f[1]['%s_Yrotation'%lft_joint], f[1]['%s_Zrotation'%lft_joint]] for f in lr.iterrows()]

                rgt_joint = lft_joint.replace('Left', 'Right')
                # rr = euler_df[[c for c in rots if rgt_joint + "_" in c]]
                # rot_order = track.skeleton[rgt_joint]['order']
                #                rgt_eulers = [[f[1]['%s_Xrotation'%rgt_joint], f[1]['%s_Yrotation'%rgt_joint], f[1]['%s_Zrotation'%rgt_joint]] for f in rr.iterrows()]

                # Create the corresponding columns in the new DataFrame
                if 'Arotation' in track.skeleton[lft_joint]['channels']:
                    new_df['%s_Arotation' % lft_joint] = -track.values[
                        '%s_Arotation' % lft_joint.replace('Left', 'Right')]
                    new_df['%s_Arotation' % lft_joint.replace('Left', 'Right')] = -track.values[
                        '%s_Arotation' % lft_joint]

                new_df['%s_Xvector' % lft_joint] = track.values[
                    '%s_Xvector' % lft_joint.replace('Left', 'Right')]
                new_df['%s_Yvector' % lft_joint] = track.values[
                    '%s_Yvector' % lft_joint.replace('Left', 'Right')]
                new_df['%s_Zvector' % lft_joint] = track.values[
                    '%s_Zvector' % lft_joint.replace('Left', 'Right')]

                new_df['%s_Xvector' % lft_joint.replace('Left', 'Right')] = track.values[
                    '%s_Xvector' % lft_joint]
                new_df['%s_Yvector' % lft_joint.replace('Left', 'Right')] = track.values[
                    '%s_Yvector' % lft_joint]
                new_df['%s_Zvector' % lft_joint.replace('Left', 'Right')] = track.values[
                    '%s_Zvector' % lft_joint]

            for joint in track.traverse():
                new_df['%s_Zvector' % joint] = -new_df['%s_Zvector' % joint]

            new_track.values = new_df
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        return X


class JointSelector_csmd(BaseEstimator, TransformerMixin):
    '''
    Allows for filtering the mocap data to include only the selected joints
    '''

    def __init__(self, joints, include_root=False):
        self.joints = joints
        self.include_root = include_root

    def fit(self, X, y=None):
        selected_joints = []
        selected_channels = []

        if self.include_root:
            print(len(X))
            selected_joints.append(X[0].root_name)

        selected_joints.extend(self.joints)

        for joint_name in selected_joints:
            selected_channels.extend([o for o in X[0].values.columns if (joint_name + "_") in o ])

        self.selected_joints = selected_joints
        self.selected_channels = selected_channels
        self.not_selected = X[0].values.columns.difference(selected_channels)
        self.not_selected_values = {c: X[0].values[c].values[0] for c in self.not_selected}

        self.orig_skeleton = X[0].skeleton
        return self

    def transform(self, X, y=None):
        print("JointSelector")
        Q = []
        for track in X:
            t2 = track.clone()
            for key in track.skeleton.keys():
                if key not in self.selected_joints:
                    t2.skeleton.pop(key)
            t2.values = track.values[self.selected_channels]

            Q.append(t2)

        return Q

    def inverse_transform(self, X, copy=None):
        Q = []
        for track in X:
            t2 = track.clone()
            t2.skeleton = self.orig_skeleton
            for d in self.not_selected:
                t2.values[d] = self.not_selected_values[d]
            Q.append(t2)

        return Q


class DerivationWorkPlace_csmd(BaseEstimator, TransformerMixin):
    '''
    Change the position work place to velocity work place
    '''

    def __init__(self):
        self.first_time = []
        self.first_pos = []

    def fit(self, X, y=None):
        self.first_time = [t.values.index[0] for t in X]
        self.first_pos = [c.values.loc[c.values.index[0]] for c in X]
        # print(type(self.first_pos[0]))
        # print(self.first_time)
        return self

    def transform(self, X, y=None):
        print("DerivationWorkPlace")
        Q = []

        for track in X:
            # print(len(track.values.index))
            first_time = track.values.index[0]
            # print(track.values.loc[track.values.index[1]]['LeftArm_Yposition'])
            list_length = len(track.values.index) - 1
            for time in range(list_length, -1, -1):
                # print(time)
                if time != 0:
                    # print(track.values.index[time])
                    track.values.loc[track.values.index[time]] = (track.values.loc[track.values.index[time]] -
                                                                  track.values.loc[
                                                                      track.values.index[time - 1]]) / track.framerate
            track.values.loc[first_time, :] = 0.0
            Q.append(track)

        return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []
        # print('input:'+len(X))

        for i in range(len(X)):
            # X[i].values = X[i].values._append(value=self.first_pos[i], index=self.first_time[i])
            # X[i] = pd.DataFrame(np.insert(X[i], self.first_time, values=self.first_pos[i], axis=0))
            # print(self.first_time[i])
            # print(self.first_pos[i])
            # base_df=pd.DataFrame({self.first_time[i]:self.first_pos[i]})
            X[i].values.loc[self.first_time[i]] = self.first_pos[i]
            # first_time = self.first_time[i]
            # first_pos = self.first_pos[i]
            list_length = len(X[i].values.index)
            # print(first_time)
            # first_df = pd.DataFrame([first_pos.values], columns=first_pos.index.tolist(), index=[first_time])
            # # print(first_df)
            # X[i].values = pd.concat([first_df, X[i].values], ignore_index=False)
            # print(X[i].values)
            for time in range(list_length):
                if time != 0:
                    X[i].values.loc[X[i].values.index[time]] = X[i].values.loc[X[i].values.index[time - 1]] + \
                                                               X[i].values.loc[X[i].values.index[time]] * X[i].framerate
                    # if time == 1:
                    #     X[i].values.loc[X[i].values.index[time]] = X[i].values.loc[first_time] + X[i].values.loc[
                    #         X[i].values.index[time]] * X[i].framerate
                    # else:
                    #     X[i].values.loc[X[i].values.index[time]] = X[i].values.loc[X[i].values.index[time - 1]] + \
                    #                                                X[i].values.loc[X[i].values.index[time]] * X[
                    #                                                    i].framerate
                    # print(X[i].values.loc[X[i].values.index[1]]['LeftArm_Yposition'])
                    # print(X[i].values['LeftForeArm_Yposition'])
                    # print(X[i].values)
                # else:
                #     X[i].values.loc[X[i].values.index[time]] = X[i].values.loc[first_time[i]] + X[i].values.loc[
                #         X[i].values.index[time]] * X[i].framerate
            Q.append(X[i])

        return Q


class LerpFirstEnd_csmd(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("LerpFirstEnd")
        Q = []

        for track in X:
            track.values = track.values.drop(index=track.values.index[0])
            track.values = track.values.drop(index=track.values.index[-1])
            Q.append(track)

        return np.array(Q)

    def inverse_transform(self, X, copy=None):
        return X


class Numpyfier_csmd(BaseEstimator, TransformerMixin):
    '''
    Just converts the values in a CSMD object into a numpy array
    Useful for the final stage of a pipeline before training
    '''

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.org_mocap_ = X[0].clone()
        self.org_mocap_.values.drop(self.org_mocap_.values.index, inplace=True)

        return self

    def transform(self, X, y=None):
        # print("Numpyfier")
        Q = []

        for track in X:
            Q.append(track.values.values)
            # print("Numpyfier:" + str(track.values.columns))

        return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            new_mocap = self.org_mocap_.clone()
            time_index = pd.to_timedelta([f for f in range(track.shape[0])], unit='s')

            new_df = pd.DataFrame(data=track, index=time_index, columns=self.org_mocap_.values.columns)

            new_mocap.values = new_df

            Q.append(new_mocap)

        return Q


class Slicer_csmd(BaseEstimator, TransformerMixin):
    '''
    Slice the data into intervals of equal size
    '''

    def __init__(self, window_size, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap
        pass

    def fit(self, X, y=None):
        self.org_mocap_ = X[0].clone()
        self.org_mocap_.values.drop(self.org_mocap_.values.index, inplace=True)

        return self

    def transform(self, X, y=None):
        print("Slicer")
        Q = []

        for track in X:
            vals = track.values.values
            nframes = vals.shape[0]
            overlap_frames = int(self.overlap * self.window_size)

            n_sequences = (nframes - overlap_frames) // (self.window_size - overlap_frames)

            if n_sequences > 0:
                y = np.zeros((n_sequences, self.window_size, vals.shape[1]))

                # extract sequences from the input data
                for i in range(0, n_sequences):
                    frameIdx = (self.window_size - overlap_frames) * i
                    Q.append(vals[frameIdx:frameIdx + self.window_size, :])

        return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            new_mocap = self.org_mocap_.clone()
            time_index = pd.to_timedelta([f for f in range(track.shape[0])], unit='s')

            new_df = pd.DataFrame(data=track, index=time_index, columns=self.org_mocap_.values.columns)

            new_mocap.values = new_df

            Q.append(new_mocap)

        return Q


class Flattener_csmd(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.concatenate(X, axis=0)


class DownSampler_csmd(BaseEstimator, TransformerMixin):
    def __init__(self, tgt_fps, keep_all=False):
        self.tgt_fps = tgt_fps
        self.keep_all = keep_all

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        Q = []

        for track in X:
            orig_fps = round(1.0 / track.framerate)
            rate = orig_fps // self.tgt_fps
            if orig_fps % self.tgt_fps != 0:
                print(
                    "error orig_fps (" + str(orig_fps) + ") is not dividable with tgt_fps (" + str(self.tgt_fps) + ")")
            # else:
            #     print("downsampling with rate: " + str(rate))

            # print(track.values.size)
            for ii in range(0, rate):
                new_track = track.clone()
                new_track.values = track.values[ii:-1:rate].copy()
                # print(new_track.values.size)
                # new_track = track[0:-1:self.rate]
                new_track.framerate = 1.0 / self.tgt_fps
                Q.append(new_track)
                if not self.keep_all:
                    break

        return Q

    def inverse_transform(self, X, copy=None):
        return X


class ReverseTime_csmd(BaseEstimator, TransformerMixin):
    def __init__(self, append=True):
        self.append = append

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        Q = []
        if self.append:
            for track in X:
                Q.append(track)
        for track in X:
            new_track = track.clone()
            new_track.values = track.values[-1::-1]
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None):
        return X


# TODO: JointsSelector (x)
# TODO: SegmentMaker
# TODO: DynamicFeaturesAdder
# TODO: ShapeFeaturesAdder
# TODO: DataFrameNumpier (x)

# class TemplateTransform(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#         return X


'''
Preprocessing Tranformers Based on sci-kit's API

By Omid Alemi
Created on June 12, 2017
'''
