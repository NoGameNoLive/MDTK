'''
-*- coding: utf-8 -*-
time: 2023/4/15 13:08
file: preprocessing.py
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


class MocapParameterizer_bvh(BaseEstimator, TransformerMixin):
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
            return X
        elif self.param_type == 'expmap':
            return self._to_expmap(X)
        elif self.param_type == 'quat':
            return X
        elif self.param_type == 'position':
            return self._to_pos(X)
        elif self.param_type == 'expmap2pos':
            return self._expmap_to_pos(X)
        else:
            raise 'param types: euler, quat, expmap, position, expmap2pos'

    #        return X

    def inverse_transform(self, X, copy=None):
        print('InverseProcess')
        if self.param_type == 'euler':
            return X
        elif self.param_type == 'expmap':
            return self._expmap_to_euler(X)
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
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            pos_df = pd.DataFrame(index=euler_df.index)

            # Copy the root rotations into the new DataFrame
            # rxp = '%s_Xrotation'%track.root_name
            # ryp = '%s_Yrotation'%track.root_name
            # rzp = '%s_Zrotation'%track.root_name
            # pos_df[rxp] = pd.Series(data=euler_df[rxp], index=pos_df.index)
            # pos_df[ryp] = pd.Series(data=euler_df[ryp], index=pos_df.index)
            # pos_df[rzp] = pd.Series(data=euler_df[rzp], index=pos_df.index)

            # List the columns that contain rotation channels
            rot_cols = [c for c in euler_df.columns if ('rotation' in c)]

            # List the columns that contain position channels
            pos_cols = [c for c in euler_df.columns if ('position' in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton)

            tree_data = {}

            for joint in track.traverse():
                # Filter end-site
                # if 'Nub' in joint:
                #     continue
                parent = track.skeleton[joint]['parent']
                rot_order = track.skeleton[joint]['order']
                # print("rot_order:" + joint + " :" + rot_order)

                # Get the rotation columns that belong to this joint
                rc = euler_df[[c for c in rot_cols if joint in c]]

                # Get the position columns that belong to this joint
                pc = euler_df[[c for c in pos_cols if joint in c]]

                # Make sure the columns are organized in xyz order
                if rc.shape[1] < 3:
                    euler_values = np.zeros((euler_df.shape[0], 3))
                    rot_order = "XYZ"
                else:
                    euler_values = np.pi / 180.0 * np.transpose(np.array(
                        [track.values['%s_%srotation' % (joint, rot_order[0])],
                         track.values['%s_%srotation' % (joint, rot_order[1])],
                         track.values['%s_%srotation' % (joint, rot_order[2])]]))

                if pc.shape[1] < 3:
                    pos_values = np.asarray([[0, 0, 0] for f in pc.iterrows()])
                else:
                    pos_values = np.asarray([[f[1]['%s_Xposition' % joint],
                                              f[1]['%s_Yposition' % joint],
                                              f[1]['%s_Zposition' % joint]] for f in pc.iterrows()])
                    # print(pos_values)

                quats = Quaternions.from_euler(np.asarray(euler_values), order=rot_order.lower(), world=False)

                # print(quats.qs)
                # print('---------------------------------------')

                tree_data[joint] = [
                    [],  # to store the rotation matrix
                    []  # to store the calculated position
                ]
                if track.root_name == joint:
                    tree_data[joint][0] = quats  # rotmats
                    # tree_data[joint][1] = np.add(pos_values, track.skeleton[joint]['offsets'])
                    tree_data[joint][1] = pos_values
                else:
                    # for every frame i, multiply this joint's rotmat to the rotmat of its parent
                    tree_data[joint][0] = tree_data[parent][0] * quats  # np.matmul(rotmats, tree_data[parent][0])

                    # add the position channel to the offset and store it in k, for every frame i
                    k = pos_values + np.asarray(track.skeleton[joint]['offsets'])

                    # print(joint)
                    # print(track.skeleton[joint]['offsets'])
                    # print(k)
                    # print(track.skeleton[joint]['children'])
                    # print('------------------------------------')

                    # multiply k to the rotmat of the parent for every frame i
                    q = tree_data[parent][0] * k  # np.matmul(k.reshape(k.shape[0],1,3), tree_data[parent][0])

                    # add q to the position of the parent, for every frame i
                    tree_data[joint][1] = tree_data[parent][1] + q  # q.reshape(k.shape[0],3) + tree_data[parent][1]
                    # if parent == 'RightLeg':
                    #     print('--------------------------')
                    #     print(track.skeleton[joint]['offsets'])
                    #     # print(tree_data[joint][0])
                    #     # print(pos_values)
                    #     print('offsets:'+str(k[0]))
                    #     print('pos:'+str(q[0]))
                    #     print('quat:'+str(tree_data[parent][0][0]))
                    #     print(tree_data[parent][0][0]*k[0])
                    #     print('euler:'+str(euler_values[0]))
                    #     print('--------------------------')

                # new Create the corresponding columns in the new DataFrame

                # flag = True
                # for e in tree_data[joint][1]:
                #     if flag:
                #         pos_df['%s_Xposition' % joint] = pd.Series(e[0])
                #         pos_df['%s_Yposition' % joint] = pd.Series(e[1])
                #         pos_df['%s_Zposition' % joint] = pd.Series(e[2])
                #         flag = False
                #         print(pos_df['%s_Zposition' % joint])
                #         continue
                #     pos_df['%s_Xposition' % joint] = pd.concat([pos_df['%s_Xposition' % joint], pd.Series(e[0])], axis=0)
                #     pos_df['%s_Yposition' % joint] = pd.concat([pos_df['%s_Yposition' % joint], pd.Series(e[0])], axis=0)
                #     pos_df['%s_Zposition' % joint] = pd.concat([pos_df['%s_Zposition' % joint], pd.Series(e[0])], axis=0)
                # pos_df['%s_Xposition' % joint].index = pos_df.index
                # pos_df['%s_Yposition' % joint].index = pos_df.index
                # pos_df['%s_Zposition' % joint].index = pos_df.index

                # data_X = pd.Series()
                # data_Y = pd.Series()
                # data_Z = pd.Series()
                # for e in tree_data[joint][1]:
                #     data_X = pd.concat([data_X, pd.Series(e[0])], axis=0)
                #     data_Y = pd.concat([data_Y, pd.Series(e[1])], axis=0)
                #     data_Z = pd.concat([data_Z, pd.Series(e[2])], axis=0)
                # # print(data_X.values)
                # pos_df['%s_Xposition' % joint] = pd.Series(data_X.values, index=pos_df.index)
                # pos_df['%s_Yposition' % joint] = pd.Series(data_Y.values, index=pos_df.index)
                # pos_df['%s_Zposition' % joint] = pd.Series(data_Z.values, index=pos_df.index)
                # if parent == 'LowerBack':
                #     print('kid_offset:' + str(track.skeleton[joint]['offsets']))
                #     print('kid_position:' + str(tree_data[joint][1][0]))
                # if joint == 'LowerBack':
                #     print('self_quat:' + str(quats[0]))
                #     print('euler:' + str(euler_values[0] * 180 / math.pi))
                #     print('all_quat:' + str(tree_data[joint][0][0]))
                #     print('parent_position:' + str(tree_data[joint][1][0]))
                # print()
                # print(pos)

                # Create the corresponding columns in the new DataFrame
                pos_df['%s_Xposition' % joint] = pd.Series(data=[e[0] for e in tree_data[joint][1]], index=pos_df.index)
                pos_df['%s_Yposition' % joint] = pd.Series(data=[e[1] for e in tree_data[joint][1]], index=pos_df.index)
                pos_df['%s_Zposition' % joint] = pd.Series(data=[e[2] for e in tree_data[joint][1]], index=pos_df.index)

            # x = []
            # y = []
            # z = []
            # for e in tree_data['RightHand'][1]:
            #     x.append(e[0]
            #              # -tree_data[track.root_name][1][0]
            #              )
            #     y.append(e[1]
            #              # -tree_data[track.root_name][1][1]
            #              )
            #     z.append(e[2]
            #              # -tree_data[track.root_name][1][2]
            #              )
            # for joint in track.traverse():
            #     for e in tree_data[joint][1]:
            #         x.append(e[0]-tree_data[track.root_name][1][0])
            #         y.append(e[1]-tree_data[track.root_name][1][1])
            #         z.append(e[2]-tree_data[track.root_name][1][2])
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # ax.scatter(x, y, z)
            # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
            # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
            # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
            # plt.show()

            new_track = track.clone()
            new_track.values = pos_df
            # print(new_track.values['LeftForeArm_Yposition'])
            # print(len(track.values.columns))
            Q.append(new_track)
        return Q

    def _pos_to_euler(self, X):
        Q = []
        for track in X:
            # print(track.values['LeftForeArm_Yposition'])
            qt = [1.0, 0.0, 0.0, 0.0]
            r_euler = {}
            channels = []
            titles = []
            # 初始化pos和euler的数值
            pos_df = track.values
            root_name = track.root_name
            # 生成Mocap的数据的行
            # rot_cols = [c for c in pos_df.columns if ('position' in c) and (root_name in c)]
            # rot_cols = rot_cols + [c.replace('position', 'rotation') for c in pos_df.columns if ('position' in c)]
            pos_cols = [c for c in pos_df.columns if ('position' in c)]
            joints = (joint for joint in track.skeleton if ('Nub' not in joint))
            euler_col = [c for c in pos_cols if ('Nub' not in c) and (track.root_name in c)]
            rot_cols = [c.replace('position', 'rotation') for c in pos_cols if ('Nub' not in c)]
            euler_col.extend(rot_cols)
            # print(euler_col)
            euler_df = pd.DataFrame(index=pos_df.index, columns=euler_col)
            # print(euler_df.columns)
            tree_data = {}
            euler_value = {}
            quats_value = {}

            for joint in joints:

                # print(joint+':'+str(track.skeleton[joint]['children']))
                parent = track.skeleton[joint]['parent']
                if parent in tree_data.keys():
                    continue
                if parent == None:
                    grand_parent = None
                else:
                    # print(grand_parent)
                    grand_parent = track.skeleton[parent]['parent']
                # print(parent)
                if parent == None:
                    continue
                rot_order = track.skeleton[joint]['order']
                # print("rot_order:" + joint + " :" + rot_order)
                children_joint = track.skeleton[joint]['children'][0]

                flag = True

                # 获取旋转值具体数据
                pc = pos_df[[c for c in pos_cols if (parent in c and 'Nub' not in c)]]
                c_pc = pos_df[[c for c in pos_cols if (joint in c and 'Nub' not in c)]]
                # print(c_pc)
                # print('-------------------------')
                if parent not in tree_data.keys():
                    # print(parent)
                    tree_data[parent] = []
                if parent not in euler_value.keys():
                    euler_value[parent] = []
                if parent not in quats_value.keys():
                    quats_value[parent] = []
                if pc.shape[1] < 3:
                    # print(parent)
                    for f in pc.iterrows():
                        euler_value[parent].append([0.0, 0.0, 0.0])
                else:
                    joint_offset = np.array(track.skeleton[joint]['offsets'])
                    # offset_norm = np.linalg.norm(joint_offset)
                    # offset_norm = 1.0
                    # normed_joint_offset = [0.0, 0.0, 0.0]
                    # if joint_offset != [0.0, 0.0, 0.0]:
                    #     for i in range(len(joint_offset)):
                    #         if offset_norm!=0:
                    #             normed_joint_offset[i] = joint_offset[i] / offset_norm
                    #         else:
                    #             normed_joint_offset[i] = joint_offset[i]

                    for f in pc.iterrows():
                        # pos_norm = math.sqrt(
                        #     (c_pc['%s_Xposition' % children_joint].loc[f[0]] - (f[1]['%s_Yposition' % joint])) ** 2 +
                        #     (c_pc['%s_Yposition' % children_joint].loc[f[0]] - (f[1]['%s_Yposition' % joint])) ** 2 +
                        #     (c_pc['%s_Zposition' % children_joint].loc[f[0]] - (f[1]['%s_Zposition' % joint])) ** 2)
                        # print(c_pc.loc[f[0]])
                        # print(f[1])
                        # print('---------------------------')
                        # pos_norm = np.linalg.norm([
                        #     (c_pc['%s_Xposition' % joint].loc[f[0]] - (f[1]['%s_Xposition' % parent])),
                        #     (c_pc['%s_Yposition' % joint].loc[f[0]] - (f[1]['%s_Yposition' % parent])),
                        #     (c_pc['%s_Zposition' % joint].loc[f[0]] - (f[1]['%s_Zposition' % parent]))
                        # ])
                        # pos_norm = 1.0

                        # if pos_norm != 0:
                        #     pos_vector = [
                        #         (c_pc['%s_Xposition' % joint].loc[f[0]]
                        #          - (f[1]['%s_Xposition' % parent])) / pos_norm,
                        #         (c_pc['%s_Yposition' % joint].loc[f[0]]
                        #          - (f[1]['%s_Yposition' % parent])) / pos_norm,
                        #         (c_pc['%s_Zposition' % joint].loc[f[0]]
                        #          - (f[1]['%s_Zposition' % parent])) / pos_norm
                        #     ]
                        # else:
                        #     pos_vector = [
                        #         (c_pc['%s_Xposition' % joint].loc[f[0]]
                        #          - (f[1]['%s_Xposition' % parent])),
                        #         (c_pc['%s_Yposition' % joint].loc[f[0]]
                        #          - (f[1]['%s_Yposition' % parent])),
                        #         (c_pc['%s_Zposition' % joint].loc[f[0]]
                        #          - (f[1]['%s_Zposition' % parent]))
                        #     ]
                        pos_vector = np.array([
                            (c_pc['%s_Xposition' % joint].loc[f[0]]
                             - (f[1]['%s_Xposition' % parent])),
                            (c_pc['%s_Yposition' % joint].loc[f[0]]
                             - (f[1]['%s_Yposition' % parent])),
                            (c_pc['%s_Zposition' % joint].loc[f[0]]
                             - (f[1]['%s_Zposition' % parent]))
                        ])
                        # print(str(joint) + '_pos:' + str(pos_value[0]))
                        # print(str(joint) + '_offset:' + str(pos_offset))
                        if np.linalg.norm(joint_offset) != 0.0:
                            # joint_offset = joint_offset / np.linalg.norm(joint_offset)
                            # pos_vector = pos_vector / np.linalg.norm(pos_vector)
                            qt = Quaternions.between(joint_offset, pos_vector)
                            # half = [pos_vector[0] + normed_joint_offset[0],
                            #         pos_vector[1] + normed_joint_offset[1],
                            #         pos_vector[2] + normed_joint_offset[2]]
                            # half = half / np.linalg.norm(half)
                            # # print(normed_joint_offset)
                            # cross_result = np.cross(normed_joint_offset, half)
                            # qt = np.array(
                            #     [np.dot(normed_joint_offset, half), cross_result[0], cross_result[1], cross_result[2]])
                            # # qt = qt / np.linalg.norm(qt)
                            # euler = qt.euler() * (180 / np.pi)
                            # euler_value[parent].append(euler[0])
                            # print(euler[0])
                            quats_value[parent].append(qt)
                        else:
                            zero = np.array([1, 0, 0])
                            qt = Quaternions.between(zero, zero)
                            quats_value[parent].append(qt)
                            continue
                        # print(cross_result)
                        # print(pos_vector)
                        # print(joint_offset)
                        # print('---------------------------')

                        # 正则化四元数
                        # qt[0] = qt[0] / np.linalg.norm(qt)
                        # qt[1] = qt[1] / np.linalg.norm(qt)
                        # qt[2] = qt[2] / np.linalg.norm(qt)

                        # print(qt)
                        # 四元数转euler
                        # print(qt)
                        # print(2 * (qt[0] * qt[1] + qt[2] * qt[3]))

                        # if parent == 'LeftShoulder':
                        #     print('joint:' + parent + ":" + str(f[1]['%s_Xposition' % parent]))
                        #     print('children:' + joint + ":" + str(
                        #         c_pc['%s_Xposition' % joint].loc[f[0]]))
                        #     print(pos_vector[0] ** 2 + pos_vector[1] ** 2 + pos_vector[2] ** 2)
                        #
                        #     print('pos_vector:' + str(pos_vector))
                        #     # print(offset_norm)
                        #     # print(pos_norm)
                        #     print('joint_vector:' + str(normed_joint_offset))
                        #     print('quaternion:' + str(qt))
                        #     print('euler:' + str(euler))
                        #     # print(np.linalg.norm(pos_vector))
                        #     # print(np.linalg.norm(normed_joint_offset))
                        #     print('--------------------------')
                        #     # flag = False

                # print(len(euler_df.index))
                # if parent == 'LowerBack':
                #     print(euler_value)
                # print(tree_data[parent])
                # print(len(quats_value[parent]))

                for i in range(len(quats_value[parent])):
                    if grand_parent != None:
                        # print(tree_data[grand_parent])
                        tree_data[parent].append(((-quats_value[grand_parent][i]) * quats_value[parent][i]).euler()[0])
                    else:
                        # print(parent)
                        tree_data[parent].append(quats_value[parent][i].euler()[0])
                # print(tree_data[parent])

                # if parent == 'LeftShoulder':
                #     print(track.skeleton[parent]['offsets'])
                #     print(tree_data[parent])

                # for t in range(len(tree_data[parent])):
                #     for e in range(len(tree_data[parent][t])):
                #         while tree_data[parent][t][e] > 180:
                #             tree_data[parent][t][e] = 360 - tree_data[parent][t][e]
                #         while tree_data[parent][t][e] < -180:
                #             tree_data[parent][t][e] = 360 + tree_data[parent][t][e]

                # if parent == 'LeftShoulder':
                #     print(tree_data[parent])

                if parent == track.root_name:
                    pos_value = [[item[1]['%s_Xposition' % parent], item[1]['%s_Yposition' % parent],
                                  item[1]['%s_Zposition' % parent]] for item in pc.iterrows()]
                    # print(pos_value)
                    pos_value_x = []
                    pos_value_y = []
                    pos_value_z = []
                    for p in pos_value:
                        pos_value_x.append(p[0])
                        pos_value_y.append(p[1])
                        pos_value_z.append(p[2])
                    euler_df['%s_Xposition' % parent] = pd.Series(data=pos_value_x, index=euler_df.index)
                    euler_df['%s_Yposition' % parent] = pd.Series(data=pos_value_y, index=euler_df.index)
                    euler_df['%s_Zposition' % parent] = pd.Series(data=pos_value_z, index=euler_df.index)
                # print(type(tree_data[parent]))
                # for e in tree_data[parent]:
                #     print(e)
                if (euler_df['%s_Xrotation' % parent].all() != 0 and (
                        euler_df['%s_Yrotation' % parent].all() != 0 and euler_df[
                    '%s_Zrotation' % parent].all() != 0)) or pd.isna(euler_df['%s_Xrotation' % parent].all() == 0):
                    euler_df['%s_Xrotation' % parent] = pd.Series(
                        data=[e[0] * (180 / np.pi) for e in tree_data[parent]],
                        index=euler_df.index)
                    euler_df['%s_Yrotation' % parent] = pd.Series(
                        data=[e[1] * (180 / np.pi) for e in tree_data[parent]],
                        index=euler_df.index)
                    euler_df['%s_Zrotation' % parent] = pd.Series(
                        data=[e[2] * (180 / np.pi) for e in tree_data[parent]],
                        index=euler_df.index)

                if 'Nub' in track.skeleton[joint]['children'][0]:
                    euler_df['%s_Xrotation' % joint] = pd.Series(data=[0.0 for e in tree_data[parent]],
                                                                 index=euler_df.index)
                    euler_df['%s_Yrotation' % joint] = pd.Series(data=[0.0 for e in tree_data[parent]],
                                                                 index=euler_df.index)
                    euler_df['%s_Zrotation' % joint] = pd.Series(data=[0.0 for e in tree_data[parent]],
                                                                 index=euler_df.index)
                # print(track.skeleton[joint]['order'])
            for column in euler_df.columns:
                if np.isnan(euler_df[column][0]):
                    euler_df[column] = 0.0
            new_track = track.clone()
            # print(track.skeleton)
            new_track.values = euler_df
            Q.append(new_track)
        return Q

    def _expmap2rot(self, expmap):

        theta = np.linalg.norm(expmap, axis=1, keepdims=True)
        nz = np.nonzero(theta)[0]

        expmap[nz, :] = expmap[nz, :] / theta[nz]

        nrows = expmap.shape[0]
        x = expmap[:, 0]
        y = expmap[:, 1]
        z = expmap[:, 2]

        s = np.sin(theta * 0.5).reshape(nrows)
        c = np.cos(theta * 0.5).reshape(nrows)

        rotmats = np.zeros((nrows, 3, 3))

        rotmats[:, 0, 0] = 2 * (x * x - 1) * s * s + 1
        rotmats[:, 0, 1] = 2 * x * y * s * s - 2 * z * c * s
        rotmats[:, 0, 2] = 2 * x * z * s * s + 2 * y * c * s
        rotmats[:, 1, 0] = 2 * x * y * s * s + 2 * z * c * s
        rotmats[:, 1, 1] = 2 * (y * y - 1) * s * s + 1
        rotmats[:, 1, 2] = 2 * y * z * s * s - 2 * x * c * s
        rotmats[:, 2, 0] = 2 * x * z * s * s - 2 * y * c * s
        rotmats[:, 2, 1] = 2 * y * z * s * s + 2 * x * c * s
        rotmats[:, 2, 2] = 2 * (z * z - 1) * s * s + 1

        return rotmats

    def _expmap_to_pos(self, X):
        '''Converts joints rotations in expmap notation to joint positions'''

        Q = []
        for track in X:
            channels = []
            titles = []
            exp_df = track.values

            # Create a new DataFrame to store the exponential map rep
            pos_df = pd.DataFrame(index=exp_df.index)

            # Copy the root rotations into the new DataFrame
            # rxp = '%s_Xrotation'%track.root_name
            # ryp = '%s_Yrotation'%track.root_name
            # rzp = '%s_Zrotation'%track.root_name
            # pos_df[rxp] = pd.Series(data=euler_df[rxp], index=pos_df.index)
            # pos_df[ryp] = pd.Series(data=euler_df[ryp], index=pos_df.index)
            # pos_df[rzp] = pd.Series(data=euler_df[rzp], index=pos_df.index)

            # List the columns that contain rotation channels
            exp_params = [c for c in exp_df.columns if
                          (any(p in c for p in ['alpha', 'beta', 'gamma']) and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton)

            tree_data = {}

            for joint in track.traverse():
                parent = track.skeleton[joint]['parent']

                if 'Nub' not in joint:
                    r = exp_df[[c for c in exp_params if joint in c]]  # Get the columns that belong to this joint
                    expmap = r.values
                    # expmap = [[f[1]['%s_alpha'%joint], f[1]['%s_beta'%joint], f[1]['%s_gamma'%joint]] for f in r.iterrows()]
                else:
                    expmap = np.zeros((exp_df.shape[0], 3))

                # Convert the eulers to rotation matrices
                # rotmats = np.asarray([Rotation(f, 'expmap').rotmat for f in expmap])
                # angs = np.linalg.norm(expmap,axis=1, keepdims=True)
                rotmats = self._expmap2rot(expmap)

                tree_data[joint] = [
                    [],  # to store the rotation matrix
                    []  # to store the calculated position
                ]
                pos_values = np.zeros((exp_df.shape[0], 3))

                if track.root_name == joint:
                    tree_data[joint][0] = rotmats
                    # tree_data[joint][1] = np.add(pos_values, track.skeleton[joint]['offsets'])
                    tree_data[joint][1] = pos_values
                else:
                    # for every frame i, multiply this joint's rotmat to the rotmat of its parent
                    tree_data[joint][0] = np.matmul(rotmats, tree_data[parent][0])

                    # add the position channel to the offset and store it in k, for every frame i
                    k = pos_values + track.skeleton[joint]['offsets']

                    # multiply k to the rotmat of the parent for every frame i
                    q = np.matmul(k.reshape(k.shape[0], 1, 3), tree_data[parent][0])

                    # add q to the position of the parent, for every frame i
                    tree_data[joint][1] = q.reshape(k.shape[0], 3) + tree_data[parent][1]

                # Create the corresponding columns in the new DataFrame
                pos_df['%s_Xposition' % joint] = pd.Series(data=tree_data[joint][1][:, 0], index=pos_df.index)
                pos_df['%s_Yposition' % joint] = pd.Series(data=tree_data[joint][1][:, 1], index=pos_df.index)
                pos_df['%s_Zposition' % joint] = pd.Series(data=tree_data[joint][1][:, 2], index=pos_df.index)

            new_track = track.clone()
            new_track.values = pos_df
            Q.append(new_track)
        return Q

    def _to_expmap(self, X):
        '''Converts Euler angles to Exponential Maps'''

        Q = []
        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            exp_df = euler_df.copy()  # pd.DataFrame(index=euler_df.index)

            # Copy the root positions into the new DataFrame
            # rxp = '%s_Xposition'%track.root_name
            # ryp = '%s_Yposition'%track.root_name
            # rzp = '%s_Zposition'%track.root_name
            # exp_df[rxp] = pd.Series(data=euler_df[rxp], index=exp_df.index)
            # exp_df[ryp] = pd.Series(data=euler_df[ryp], index=exp_df.index)
            # exp_df[rzp] = pd.Series(data=euler_df[rzp], index=exp_df.index)

            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                # print(joint)
                r = euler_df[[c for c in rots if joint in c]]  # Get the columns that belong to this joint
                rot_order = track.skeleton[joint]['order']
                r1_col = '%s_%srotation' % (joint, rot_order[0])
                r2_col = '%s_%srotation' % (joint, rot_order[1])
                r3_col = '%s_%srotation' % (joint, rot_order[2])

                exp_df.drop([r1_col, r2_col, r3_col], axis=1, inplace=True)
                euler = [[f[1][r1_col], f[1][r2_col], f[1][r3_col]] for f in r.iterrows()]
                # exps = [Rotation(f, 'euler', from_deg=True, order=rot_order).to_expmap() for f in euler] # Convert the eulers to exp maps
                exps = unroll(
                    np.array([euler2expmap(f, rot_order, True) for f in euler]))  # Convert the exp maps to eulers
                # exps = euler2expmap2(euler, rot_order, True) # Convert the eulers to exp maps

                # Create the corresponding columns in the new DataFrame

                exp_df.insert(loc=0, column='%s_gamma' % joint,
                              value=pd.Series(data=[e[2] for e in exps], index=exp_df.index))
                exp_df.insert(loc=0, column='%s_beta' % joint,
                              value=pd.Series(data=[e[1] for e in exps], index=exp_df.index))
                exp_df.insert(loc=0, column='%s_alpha' % joint,
                              value=pd.Series(data=[e[0] for e in exps], index=exp_df.index))

            # print(exp_df.columns)
            new_track = track.clone()
            new_track.values = exp_df
            Q.append(new_track)

        return Q

    def _expmap_to_euler(self, X):
        Q = []
        for track in X:
            channels = []
            titles = []
            exp_df = track.values

            # Create a new DataFrame to store the exponential map rep
            # euler_df = pd.DataFrame(index=exp_df.index)
            euler_df = exp_df.copy()

            # Copy the root positions into the new DataFrame
            # rxp = '%s_Xposition'%track.root_name
            # ryp = '%s_Yposition'%track.root_name
            # rzp = '%s_Zposition'%track.root_name
            # euler_df[rxp] = pd.Series(data=exp_df[rxp], index=euler_df.index)
            # euler_df[ryp] = pd.Series(data=exp_df[ryp], index=euler_df.index)
            # euler_df[rzp] = pd.Series(data=exp_df[rzp], index=euler_df.index)

            # List the columns that contain rotation channels
            exp_params = [c for c in exp_df.columns if
                          (any(p in c for p in ['alpha', 'beta', 'gamma']) and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = exp_df[[c for c in exp_params if joint in c]]  # Get the columns that belong to this joint

                euler_df.drop(['%s_alpha' % joint, '%s_beta' % joint, '%s_gamma' % joint], axis=1, inplace=True)
                expmap = [[f[1]['%s_alpha' % joint], f[1]['%s_beta' % joint], f[1]['%s_gamma' % joint]] for f in
                          r.iterrows()]  # Make sure the columsn are organized in xyz order
                rot_order = track.skeleton[joint]['order']
                # euler_rots = [Rotation(f, 'expmap').to_euler(True, rot_order) for f in expmap] # Convert the exp maps to eulers
                euler_rots = [expmap2euler(f, rot_order, True) for f in expmap]  # Convert the exp maps to eulers

                # Create the corresponding columns in the new DataFrame

                euler_df['%s_%srotation' % (joint, rot_order[0])] = pd.Series(data=[e[0] for e in euler_rots],
                                                                              index=euler_df.index)
                euler_df['%s_%srotation' % (joint, rot_order[1])] = pd.Series(data=[e[1] for e in euler_rots],
                                                                              index=euler_df.index)
                euler_df['%s_%srotation' % (joint, rot_order[2])] = pd.Series(data=[e[2] for e in euler_rots],
                                                                              index=euler_df.index)

            new_track = track.clone()
            new_track.values = euler_df
            Q.append(new_track)

        return Q


class Mirror_bvh(BaseEstimator, TransformerMixin):
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

            if self.axis == "X":
                signs = np.array([1, -1, -1])
            if self.axis == "Y":
                signs = np.array([-1, 1, -1])
            if self.axis == "Z":
                signs = np.array([-1, -1, 1])

            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            new_df = pd.DataFrame(index=euler_df.index)

            # Copy the root positions into the new DataFrame
            rxp = '%s_Xposition' % track.root_name
            ryp = '%s_Yposition' % track.root_name
            rzp = '%s_Zposition' % track.root_name
            new_df[rxp] = pd.Series(data=-signs[0] * euler_df[rxp], index=new_df.index)
            new_df[ryp] = pd.Series(data=-signs[1] * euler_df[ryp], index=new_df.index)
            new_df[rzp] = pd.Series(data=-signs[2] * euler_df[rzp], index=new_df.index)

            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]
            # lft_rots = [c for c in euler_df.columns if ('Left' in c and 'rotation' in c and 'Nub' not in c)]
            # rgt_rots = [c for c in euler_df.columns if ('Right' in c and 'rotation' in c and 'Nub' not in c)]
            lft_joints = (joint for joint in track.skeleton if 'Left' in joint and 'Nub' not in joint)
            rgt_joints = (joint for joint in track.skeleton if 'Right' in joint and 'Nub' not in joint)

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

                new_df['%s_Xrotation' % lft_joint] = pd.Series(data=signs[0] * track.values['%s_Xrotation' % rgt_joint],
                                                               index=new_df.index)
                new_df['%s_Yrotation' % lft_joint] = pd.Series(data=signs[1] * track.values['%s_Yrotation' % rgt_joint],
                                                               index=new_df.index)
                new_df['%s_Zrotation' % lft_joint] = pd.Series(data=signs[2] * track.values['%s_Zrotation' % rgt_joint],
                                                               index=new_df.index)

                new_df['%s_Xrotation' % rgt_joint] = pd.Series(data=signs[0] * track.values['%s_Xrotation' % lft_joint],
                                                               index=new_df.index)
                new_df['%s_Yrotation' % rgt_joint] = pd.Series(data=signs[1] * track.values['%s_Yrotation' % lft_joint],
                                                               index=new_df.index)
                new_df['%s_Zrotation' % rgt_joint] = pd.Series(data=signs[2] * track.values['%s_Zrotation' % lft_joint],
                                                               index=new_df.index)

            # List the joints that are not left or right, i.e. are on the trunk
            joints = (joint for joint in track.skeleton if
                      'Nub' not in joint and 'Left' not in joint and 'Right' not in joint)

            for joint in joints:
                # r = euler_df[[c for c in rots if joint in c]] # Get the columns that belong to this joint
                # rot_order = track.skeleton[joint]['order']

                # eulers = [[f[1]['%s_Xrotation'%joint], f[1]['%s_Yrotation'%joint], f[1]['%s_Zrotation'%joint]] for f in r.iterrows()]

                # Create the corresponding columns in the new DataFrame
                new_df['%s_Xrotation' % joint] = pd.Series(data=signs[0] * track.values['%s_Xrotation' % joint],
                                                           index=new_df.index)
                new_df['%s_Yrotation' % joint] = pd.Series(data=signs[1] * track.values['%s_Yrotation' % joint],
                                                           index=new_df.index)
                new_df['%s_Zrotation' % joint] = pd.Series(data=signs[2] * track.values['%s_Zrotation' % joint],
                                                           index=new_df.index)

            new_track.values = new_df
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        return X


class EulerReorder_bvh(BaseEstimator, TransformerMixin):
    def __init__(self, new_order):
        """
        Add a
        """
        self.new_order = new_order

    def fit(self, X, y=None):
        self.orig_skeleton = copy.deepcopy(X[0].skeleton)
        print(self.orig_skeleton)
        return self

    def transform(self, X, y=None):
        Q = []

        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            new_df = pd.DataFrame(index=euler_df.index)

            # Copy the root positions into the new DataFrame
            rxp = '%s_Xposition' % track.root_name
            ryp = '%s_Yposition' % track.root_name
            rzp = '%s_Zposition' % track.root_name
            new_df[rxp] = pd.Series(data=euler_df[rxp], index=new_df.index)
            new_df[ryp] = pd.Series(data=euler_df[ryp], index=new_df.index)
            new_df[rzp] = pd.Series(data=euler_df[rzp], index=new_df.index)

            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            new_track = track.clone()
            for joint in joints:
                r = euler_df[[c for c in rots if joint in c]]  # Get the columns that belong to this joint
                rot_order = track.skeleton[joint]['order']

                euler = [
                    [f[1]['%s_Xrotation' % (joint)], f[1]['%s_Yrotation' % (joint)], f[1]['%s_Zrotation' % (joint)]] for
                    f in r.iterrows()]
                new_euler = [euler_reorder(f, rot_order, self.new_order, True) for f in euler]
                # new_euler = euler_reorder2(np.array(euler), rot_order, self.new_order, True)

                # Create the corresponding columns in the new DataFrame
                new_df['%s_%srotation' % (joint, self.new_order[0])] = pd.Series(data=[e[0] for e in new_euler],
                                                                                 index=new_df.index)
                new_df['%s_%srotation' % (joint, self.new_order[1])] = pd.Series(data=[e[1] for e in new_euler],
                                                                                 index=new_df.index)
                new_df['%s_%srotation' % (joint, self.new_order[2])] = pd.Series(data=[e[2] for e in new_euler],
                                                                                 index=new_df.index)

                new_track.skeleton[joint]['order'] = self.new_order

            new_track.values = new_df
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        return X


class JointSelector_bvh(BaseEstimator, TransformerMixin):
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
            selected_channels.extend([o for o in X[0].values.columns if (joint_name + "_") in o and 'Nub' not in o])

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


class RelativePositionWorkPlace_bvh(BaseEstimator, TransformerMixin):
    '''
    Change Position to Relative position
    '''

    def __init__(self, change_space=True):
        self.if_change_space = change_space
        self.trans_quat = []

    def fit(self, X, y=None):
        # for track in X:
        #     if self.if_change_space:
        #         right_up_leg_projection = [track.values['RightUpLeg_Xposition'], 0,
        #                                    track.values['RightUpLeg_Zposition']]
        #         change_matrix = []
        right_up_leg_projection = {}
        target_vector = np.array([0, 0, 1.0])
        proj_qt = []
        for track in X:
            for frame in track.values.index:
                right_up_leg_projection[frame] = np.array([track.values['RightUpLeg_Xposition'].loc[frame], 0,
                                                           track.values['RightUpLeg_Zposition'].loc[frame]])
            for tick in right_up_leg_projection.keys():
                quat = Quaternions.between(right_up_leg_projection[tick], target_vector)
                proj_qt.append(quat)
            self.trans_quat.append(proj_qt)
        return self

    def transform(self, X, y=None):
        # print("RelativePositionWorkPlace")
        Q = []
        cont = 0
        for track in X:
            # print(self.trans_quat[cont])
            for joint in track.traverse():
                if joint != track.root_name:
                    track.values['%s_Xposition' % (joint)] -= track.values['%s_Xposition' % (track.root_name)]
                    track.values['%s_Yposition' % (joint)] -= track.values['%s_Yposition' % (track.root_name)]
                    track.values['%s_Zposition' % (joint)] -= track.values['%s_Zposition' % (track.root_name)]
                    # for time in track.values.index:
                    #     track.values['%s_Xposition' % (joint)].loc[time] -= \
                    #     track.values['%s_Xposition' % (track.root_name)].loc[time]
                    #     track.values['%s_Yposition' % (joint)].loc[time] -= \
                    #     track.values['%s_Yposition' % (track.root_name)].loc[time]
                    #     track.values['%s_Zposition' % (joint)].loc[time] -= \
                    #     track.values['%s_Zposition' % (track.root_name)].loc[time]

                if self.if_change_space:
                    if joint != track.root_name:
                        time = 0
                        for tick in track.values.index:
                            temp_vector = np.array(
                                [track.values['%s_Xposition' % joint].loc[tick],
                                 track.values['%s_Yposition' % joint].loc[tick],
                                 track.values['%s_Zposition' % joint].loc[tick]])
                            # print(self.trans_quat[cont])
                            temp_vector = (self.trans_quat[cont][time] * temp_vector)[0]
                            time += 1
                            track.values['%s_Xposition' % joint].loc[tick] = temp_vector[0]
                            track.values['%s_Yposition' % joint].loc[tick] = temp_vector[1]
                            track.values['%s_Zposition' % joint].loc[tick] = temp_vector[2]
            # print("Numpyfier:" + str(track.values.columns))
            # print(cont)
            # print(track.values)
            Q.append(track)
        return Q

    def inverse_transform(self, X, copy=None):
        Q = []

        cont = -1

        # print(len(X))
        for track in X:
            cont += 1
            # print((-self.trans_quat[cont]))
            for joint in track.traverse():
                if joint != track.root_name:
                    track.values['%s_Xposition' % joint] += track.values['%s_Xposition' % track.root_name]
                    track.values['%s_Yposition' % joint] += track.values['%s_Yposition' % track.root_name]
                    track.values['%s_Zposition' % joint] += track.values['%s_Zposition' % track.root_name]

                if self.if_change_space:
                    if joint != track.root_name:
                        time = 0
                        for tick in track.values.index:
                            temp_vector = np.array(
                                [track.values['%s_Xposition' % joint].loc[tick],
                                 track.values['%s_Yposition' % joint].loc[tick],
                                 track.values['%s_Zposition' % joint].loc[tick]])
                            temp_vector = ((-self.trans_quat[cont][time]) * temp_vector)[0]
                            time += 1
                            track.values['%s_Xposition' % joint].loc[tick] = temp_vector[0]
                            track.values['%s_Yposition' % joint].loc[tick] = temp_vector[1]
                            track.values['%s_Zposition' % joint].loc[tick] = temp_vector[2]
            # print(track.values)
            Q.append(track)

        return Q


class DerivationWorkPlace_bvh(BaseEstimator, TransformerMixin):
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


class LerpFirstEnd_bvh(BaseEstimator, TransformerMixin):
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

class LerpNub_bvh(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # print("LerpNub")
        Q = []

        for track in X:
            for column in track.values.columns:
                if 'Nub' in column:
                    track.values = track.values.drop(columns=column)
            Q.append(track)

        return np.array(Q)

    def inverse_transform(self, X, copy=None):
        return X


class Numpyfier_bvh(BaseEstimator, TransformerMixin):
    '''
    Just converts the values in a MocapData object into a numpy array
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


class Slicer_bvh(BaseEstimator, TransformerMixin):
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


class RootTransformer_bvh(BaseEstimator, TransformerMixin):
    def __init__(self, method, position_smoothing=0, rotation_smoothing=0):
        """
        Accepted methods:
            abdolute_translation_deltas
            pos_rot_deltas
        """
        self.method = method
        self.position_smoothing = position_smoothing
        self.rotation_smoothing = rotation_smoothing

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("RootTransformer")
        Q = []

        for track in X:
            if self.method == 'abdolute_translation_deltas':
                new_df = track.values.copy()
                xpcol = '%s_Xposition' % track.root_name
                ypcol = '%s_Yposition' % track.root_name
                zpcol = '%s_Zposition' % track.root_name

                dxpcol = '%s_dXposition' % track.root_name
                dzpcol = '%s_dZposition' % track.root_name

                x = track.values[xpcol].copy()
                z = track.values[zpcol].copy()

                if self.position_smoothing > 0:
                    x_sm = filters.gaussian_filter1d(x, self.position_smoothing, axis=0, mode='nearest')
                    z_sm = filters.gaussian_filter1d(z, self.position_smoothing, axis=0, mode='nearest')
                    dx = pd.Series(data=x_sm, index=new_df.index).diff()
                    dz = pd.Series(data=z_sm, index=new_df.index).diff()
                    new_df[xpcol] = x - x_sm
                    new_df[zpcol] = z - z_sm
                else:
                    dx = x.diff()
                    dz = z.diff()
                    new_df.drop([xpcol, zpcol], axis=1, inplace=True)

                dx[0] = dx[1]
                dz[0] = dz[1]

                new_df[dxpcol] = dx
                new_df[dzpcol] = dz

                new_track = track.clone()
                new_track.values = new_df
            # end of abdolute_translation_deltas

            elif self.method == 'pos_rot_deltas':
                new_track = track.clone()

                # Absolute columns
                xp_col = '%s_Xposition' % track.root_name
                yp_col = '%s_Yposition' % track.root_name
                zp_col = '%s_Zposition' % track.root_name

                # rot_order = track.skeleton[track.root_name]['order']
                # %(joint, rot_order[0])

                rot_order = track.skeleton[track.root_name]['order']
                r1_col = '%s_%srotation' % (track.root_name, rot_order[0])
                r2_col = '%s_%srotation' % (track.root_name, rot_order[1])
                r3_col = '%s_%srotation' % (track.root_name, rot_order[2])

                # Delta columns
                dxp_col = '%s_dXposition' % track.root_name
                dzp_col = '%s_dZposition' % track.root_name

                dxr_col = '%s_dXrotation' % track.root_name
                dyr_col = '%s_dYrotation' % track.root_name
                dzr_col = '%s_dZrotation' % track.root_name

                positions = np.transpose(np.array([track.values[xp_col], track.values[yp_col], track.values[zp_col]]))
                rotations = np.pi / 180.0 * np.transpose(
                    np.array([track.values[r1_col], track.values[r2_col], track.values[r3_col]]))

                """ Get Trajectory and smooth it"""
                trajectory_filterwidth = self.position_smoothing
                reference = positions.copy() * np.array([1, 0, 1])
                if trajectory_filterwidth > 0:
                    reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')

                """ Get Root Velocity """
                velocity = np.diff(reference, axis=0)
                # print(velocity.shape)
                # print(velocity.shape)
                # print(velocity[0, :].shape)
                velocity = np.vstack((velocity[0, :], velocity))

                """ Remove Root Translation """
                positions = positions - reference

                """ Get Forward Direction along the x-z plane, assuming character is facig z-forward """
                # forward = [Rotation(f, 'euler', from_deg=True, order=rot_order).rotmat[:,2] for f in rotations] # get the z-axis of the rotation matrix, assuming character is facig z-forward
                # print("order:" + rot_order.lower())
                quats = Quaternions.from_euler(rotations, order=rot_order.lower(), world=False)
                forward = quats * np.array([[0, 0, 1]])
                forward[:, 1] = 0

                """ Smooth Forward Direction """
                direction_filterwidth = self.rotation_smoothing
                if direction_filterwidth > 0:
                    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')

                forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

                """ Remove Y Rotation """
                target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
                rotation = Quaternions.between(target, forward)[:, np.newaxis]
                positions = (-rotation[:, 0]) * positions
                new_rotations = (-rotation[:, 0]) * quats

                """ Get Root Rotation """
                # print(rotation[:,0])
                velocity = (-rotation[:, 0]) * velocity
                rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
                rvelocity = np.vstack((rvelocity[0], rvelocity))

                eulers = np.array([t3d.euler.quat2euler(q, axes=('s' + rot_order.lower()[::-1]))[::-1] for q in
                                   new_rotations]) * 180.0 / np.pi

                new_df = track.values.copy()

                root_pos_x = pd.Series(data=positions[:, 0], index=new_df.index)
                root_pos_y = pd.Series(data=positions[:, 1], index=new_df.index)
                root_pos_z = pd.Series(data=positions[:, 2], index=new_df.index)
                root_pos_x_diff = pd.Series(data=velocity[:, 0], index=new_df.index)
                root_pos_z_diff = pd.Series(data=velocity[:, 2], index=new_df.index)

                root_rot_1 = pd.Series(data=eulers[:, 0], index=new_df.index)
                root_rot_2 = pd.Series(data=eulers[:, 1], index=new_df.index)
                root_rot_3 = pd.Series(data=eulers[:, 2], index=new_df.index)
                root_rot_y_diff = pd.Series(data=rvelocity[:, 0], index=new_df.index)

                # new_df.drop([xr_col, yr_col, zr_col, xp_col, zp_col], axis=1, inplace=True)

                new_df[xp_col] = root_pos_x
                new_df[yp_col] = root_pos_y
                new_df[zp_col] = root_pos_z
                new_df[dxp_col] = root_pos_x_diff
                new_df[dzp_col] = root_pos_z_diff

                new_df[r1_col] = root_rot_1
                new_df[r2_col] = root_rot_2
                new_df[r3_col] = root_rot_3
                # new_df[dxr_col] = root_rot_x_diff
                new_df[dyr_col] = root_rot_y_diff
                # new_df[dzr_col] = root_rot_z_diff

                new_track.values = new_df


            elif self.method == 'hip_centric':
                new_track = track.clone()

                # Absolute columns
                xp_col = '%s_Xposition' % track.root_name
                yp_col = '%s_Yposition' % track.root_name
                zp_col = '%s_Zposition' % track.root_name

                xr_col = '%s_Xrotation' % track.root_name
                yr_col = '%s_Yrotation' % track.root_name
                zr_col = '%s_Zrotation' % track.root_name

                new_df = track.values.copy()

                all_zeros = np.zeros(track.values[xp_col].values.shape)

                new_df[xp_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[yp_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[zp_col] = pd.Series(data=all_zeros, index=new_df.index)

                new_df[xr_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[yr_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[zr_col] = pd.Series(data=all_zeros, index=new_df.index)

                new_track.values = new_df

            # print(new_track.values.columns)
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        Q = []

        # TODO: simplify this implementation

        startx = 0
        startz = 0

        if start_pos is not None:
            startx, startz = start_pos

        for track in X:
            new_track = track.clone()
            if self.method == 'abdolute_translation_deltas':
                new_df = new_track.values
                xpcol = '%s_Xposition' % track.root_name
                ypcol = '%s_Yposition' % track.root_name
                zpcol = '%s_Zposition' % track.root_name

                dxpcol = '%s_dXposition' % track.root_name
                dzpcol = '%s_dZposition' % track.root_name

                dx = track.values[dxpcol].values
                dz = track.values[dzpcol].values

                recx = [startx]
                recz = [startz]

                for i in range(dx.shape[0] - 1):
                    recx.append(recx[i] + dx[i + 1])
                    recz.append(recz[i] + dz[i + 1])

                # recx = [recx[i]+dx[i+1] for i in range(dx.shape[0]-1)]
                # recz = [recz[i]+dz[i+1] for i in range(dz.shape[0]-1)]
                # recx = dx[:-1] + dx[1:]
                # recz = dz[:-1] + dz[1:]
                if self.position_smoothing > 0:
                    new_df[xpcol] = pd.Series(data=new_df[xpcol] + recx, index=new_df.index)
                    new_df[zpcol] = pd.Series(data=new_df[zpcol] + recz, index=new_df.index)
                else:
                    new_df[xpcol] = pd.Series(data=recx, index=new_df.index)
                    new_df[zpcol] = pd.Series(data=recz, index=new_df.index)

                new_df.drop([dxpcol, dzpcol], axis=1, inplace=True)

                new_track.values = new_df
            # end of abdolute_translation_deltas

            elif self.method == 'pos_rot_deltas':
                # Absolute columns
                rot_order = track.skeleton[track.root_name]['order']
                xp_col = '%s_Xposition' % track.root_name
                yp_col = '%s_Yposition' % track.root_name
                zp_col = '%s_Zposition' % track.root_name

                xr_col = '%s_Xrotation' % track.root_name
                yr_col = '%s_Yrotation' % track.root_name
                zr_col = '%s_Zrotation' % track.root_name
                r1_col = '%s_%srotation' % (track.root_name, rot_order[0])
                r2_col = '%s_%srotation' % (track.root_name, rot_order[1])
                r3_col = '%s_%srotation' % (track.root_name, rot_order[2])

                # Delta columns
                dxp_col = '%s_dXposition' % track.root_name
                dzp_col = '%s_dZposition' % track.root_name

                dyr_col = '%s_dYrotation' % track.root_name

                positions = np.transpose(np.array([track.values[xp_col], track.values[yp_col], track.values[zp_col]]))
                rotations = np.pi / 180.0 * np.transpose(
                    np.array([track.values[r1_col], track.values[r2_col], track.values[r3_col]]))
                quats = Quaternions.from_euler(rotations, order=rot_order.lower(), world=False)

                new_df = track.values.copy()

                dx = track.values[dxp_col].values
                dz = track.values[dzp_col].values

                dry = track.values[dyr_col].values

                # rec_p = np.array([startx, 0, startz])+positions[0,:]
                rec_ry = Quaternions.id(quats.shape[0])
                rec_xp = [0]
                rec_zp = [0]

                # rec_r = Quaternions.id(quats.shape[0])

                for i in range(dx.shape[0] - 1):
                    # print(dry[i])
                    q_y = Quaternions.from_angle_axis(np.array(dry[i + 1]), np.array([0, 1, 0]))
                    rec_ry[i + 1] = q_y * rec_ry[i]
                    # print("dx: + " + str(dx[i+1]))
                    dp = rec_ry[i + 1] * np.array([dx[i + 1], 0, dz[i + 1]])
                    rec_xp.append(rec_xp[i] + dp[0, 0])
                    rec_zp.append(rec_zp[i] + dp[0, 2])

                rec_r = rec_ry * quats
                pp = rec_ry * positions
                rec_xp = rec_xp + pp[:, 0]
                rec_zp = rec_zp + pp[:, 2]

                eulers = np.array([t3d.euler.quat2euler(q, axes=('s' + rot_order.lower()[::-1]))[::-1] for q in
                                   rec_r]) * 180.0 / np.pi

                new_df = track.values.copy()

                root_rot_1 = pd.Series(data=eulers[:, 0], index=new_df.index)
                root_rot_2 = pd.Series(data=eulers[:, 1], index=new_df.index)
                root_rot_3 = pd.Series(data=eulers[:, 2], index=new_df.index)

                new_df[xp_col] = pd.Series(data=rec_xp, index=new_df.index)
                new_df[zp_col] = pd.Series(data=rec_zp, index=new_df.index)

                new_df[r1_col] = pd.Series(data=root_rot_1, index=new_df.index)
                new_df[r2_col] = pd.Series(data=root_rot_2, index=new_df.index)
                new_df[r3_col] = pd.Series(data=root_rot_3, index=new_df.index)

                new_df.drop([dyr_col, dxp_col, dzp_col], axis=1, inplace=True)

                new_track.values = new_df

            # print(new_track.values.columns)
            Q.append(new_track)

        return Q


class RootCentricPositionNormalizer_bvh(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Q = []

        for track in X:
            new_track = track.clone()

            rxp = '%s_Xposition' % track.root_name
            ryp = '%s_Yposition' % track.root_name
            rzp = '%s_Zposition' % track.root_name

            projected_root_pos = track.values[[rxp, ryp, rzp]]

            projected_root_pos.loc[:, ryp] = 0  # we want the root's projection on the floor plane as the ref

            new_df = pd.DataFrame(index=track.values.index)

            all_but_root = [joint for joint in track.skeleton if track.root_name not in joint]
            # all_but_root = [joint for joint in track.skeleton]
            for joint in all_but_root:
                new_df['%s_Xposition' % joint] = pd.Series(
                    data=track.values['%s_Xposition' % joint] - projected_root_pos[rxp], index=new_df.index)
                new_df['%s_Yposition' % joint] = pd.Series(
                    data=track.values['%s_Yposition' % joint] - projected_root_pos[ryp], index=new_df.index)
                new_df['%s_Zposition' % joint] = pd.Series(
                    data=track.values['%s_Zposition' % joint] - projected_root_pos[rzp], index=new_df.index)

            # keep the root as it is now
            new_df[rxp] = track.values[rxp]
            new_df[ryp] = track.values[ryp]
            new_df[rzp] = track.values[rzp]

            new_track.values = new_df

            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            new_track = track.clone()

            rxp = '%s_Xposition' % track.root_name
            ryp = '%s_Yposition' % track.root_name
            rzp = '%s_Zposition' % track.root_name

            projected_root_pos = track.values[[rxp, ryp, rzp]]

            projected_root_pos.loc[:, ryp] = 0  # we want the root's projection on the floor plane as the ref

            new_df = pd.DataFrame(index=track.values.index)

            for joint in track.skeleton:
                new_df['%s_Xposition' % joint] = pd.Series(
                    data=track.values['%s_Xposition' % joint] + projected_root_pos[rxp], index=new_df.index)
                new_df['%s_Yposition' % joint] = pd.Series(
                    data=track.values['%s_Yposition' % joint] + projected_root_pos[ryp], index=new_df.index)
                new_df['%s_Zposition' % joint] = pd.Series(
                    data=track.values['%s_Zposition' % joint] + projected_root_pos[rzp], index=new_df.index)

            new_track.values = new_df

            Q.append(new_track)

        return Q


class Flattener_bvh(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.concatenate(X, axis=0)


class ConstantsRemover_bvh(BaseEstimator, TransformerMixin):
    '''
    For now it just looks at the first track
    '''

    def __init__(self, eps=1e-6):
        self.eps = eps

    def fit(self, X, y=None):
        stds = X[0].values.std()
        cols = X[0].values.columns.values
        self.const_dims_ = [c for c in cols if (stds[c] < self.eps).any()]
        # print(self.const_dims_)
        self.const_values_ = {c: X[0].values[c].values[0] for c in cols if (stds[c] < self.eps).any()}
        # print(self.const_values_)
        return self

    def transform(self, X, y=None):
        Q = []

        for track in X:
            t2 = track.clone()
            # for key in t2.skeleton.keys():
            #    if key in self.ConstDims_:
            #        t2.skeleton.pop(key)
            # print(track.values.columns.difference(self.const_dims_))
            t2.values.drop(self.const_dims_, axis=1, inplace=True)
            # t2.values = track.values[track.values.columns.difference(self.const_dims_)]
            Q.append(t2)

        return Q

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            t2 = track.clone()
            for d in self.const_dims_:
                t2.values[d] = self.const_values_[d]
            #                t2.values.assign(d=pd.Series(data=self.const_values_[d], index = t2.values.index))
            Q.append(t2)

        return Q


class ListStandardScaler_bvh(BaseEstimator, TransformerMixin):
    def __init__(self, is_DataFrame=False):
        self.is_DataFrame = is_DataFrame

    def fit(self, X, y=None):
        if self.is_DataFrame:
            X_train_flat = np.concatenate([m.values for m in X], axis=0)
        else:
            X_train_flat = np.concatenate([m for m in X], axis=0)

        self.data_mean_ = np.mean(X_train_flat, axis=0)
        self.data_std_ = np.std(X_train_flat, axis=0)

        return self

    def transform(self, X, y=None):
        Q = []

        for track in X:
            if self.is_DataFrame:
                normalized_track = track.copy()
                normalized_track.values = (track.values - self.data_mean_) / self.data_std_
            else:
                normalized_track = (track - self.data_mean_) / self.data_std_

            Q.append(normalized_track)

        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:

            if self.is_DataFrame:
                unnormalized_track = track.copy()
                unnormalized_track.values = (track.values * self.data_std_) + self.data_mean_
            else:
                unnormalized_track = (track * self.data_std_) + self.data_mean_

            Q.append(unnormalized_track)

        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)


class ListMinMaxScaler_bvh(BaseEstimator, TransformerMixin):
    def __init__(self, is_DataFrame=False):
        self.is_DataFrame = is_DataFrame

    def fit(self, X, y=None):
        if self.is_DataFrame:
            X_train_flat = np.concatenate([m.values for m in X], axis=0)
        else:
            X_train_flat = np.concatenate([m for m in X], axis=0)

        self.data_max_ = np.max(X_train_flat, axis=0)
        self.data_min_ = np.min(X_train_flat, axis=0)

        return self

    def transform(self, X, y=None):
        Q = []

        for track in X:
            if self.is_DataFrame:
                normalized_track = track.copy()
                normalized_track.values = (track.values - self.data_min_) / (self.data_max_ - self.data_min_)
            else:
                normalized_track = (track - self.data_min_) / (self.data_max_ - self.data_min_)

            Q.append(normalized_track)

        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:

            if self.is_DataFrame:
                unnormalized_track = track.copy()
                unnormalized_track.values = (track.values * (self.data_max_ - self.data_min_)) + self.data_min_
            else:
                unnormalized_track = (track * (self.data_max_ - self.data_min_)) + self.data_min_

            Q.append(unnormalized_track)

        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)


class DownSampler_bvh(BaseEstimator, TransformerMixin):
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


class ReverseTime_bvh(BaseEstimator, TransformerMixin):
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

#
# '''
# Preprocessing Tranformers Based on sci-kit's API
#
# By Omid Alemi
# Created on June 12, 2017
# '''
