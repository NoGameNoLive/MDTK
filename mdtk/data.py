'''
-*- coding: utf-8 -*-
time: 2023/4/15 13:04
file: data.py
author: Endy_Liu_Noonell
'''

import numpy as np


class Joint():
    def __init__(self, name, parent=None, children=None):
        self.name = name
        self.parent = parent
        self.children = children


class MocapData():
    def __init__(self):
        self.skeleton = {}
        self.values = None
        self.channel_names = []
        self.framerate = 0.0
        self.root_name = ''

    def traverse(self, j=None):
        stack = [self.root_name]
        while stack:
            joint = stack.pop()
            yield joint
            for c in self.skeleton[joint]['children']:
                stack.append(c)

    def clone(self):
        import copy
        new_data = MocapData()
        new_data.skeleton = copy.deepcopy(self.skeleton)
        new_data.values = copy.deepcopy(self.values)
        new_data.channel_names = copy.deepcopy(self.channel_names)
        new_data.root_name = copy.deepcopy(self.root_name)
        new_data.framerate = copy.deepcopy(self.framerate)
        return new_data

    def get_all_channels(self):
        '''Returns all of the channels parsed from the file as a 2D numpy array'''

        frames = [f[1] for f in self.values]
        return np.asarray([[channel[2] for channel in frame] for frame in frames])

    def get_skeleton_tree(self):
        tree = []
        root_key = [j for j in self.skeleton if self.skeleton[j]['parent'] == None][0]

        root_joint = Joint(root_key)

    def get_empty_channels(self):
        # TODO
        pass

    def get_constant_channels(self):
        # TODO
        pass


class CSMD():
    def __init__(self):
        self.skeleton = {}
        self.values = None
        self.channel_names = []
        self.time_list = []
        self.root_name = ''
        self.endsite_list = []
        self.root_position = []
        self.root_direction = []

    def traverse(self, j=None):
        stack = [self.root_name]
        while stack:
            joint = stack.pop()
            yield joint
            for c in self.skeleton[joint]['children']:
                stack.append(c)

    def clone(self):
        import copy
        new_data = CSMD()
        new_data.skeleton = copy.deepcopy(self.skeleton)
        new_data.values = copy.deepcopy(self.values)
        new_data.channel_names = copy.deepcopy(self.channel_names)
        new_data.root_name = copy.deepcopy(self.root_name)
        new_data.time_list = copy.deepcopy(self.time_list)
        new_data.root_name = copy.deepcopy(self.root_name)
        new_data.root_position=copy.deepcopy(self.root_position)
        new_data.root_direction = copy.deepcopy(self.root_direction)
        new_data.endsite_list = copy.deepcopy(self.endsite_list)
        new_data.time_list = copy.deepcopy(self.root_position)
        new_data.time_list = copy.deepcopy(self.root_direction)
        return new_data

    def get_all_channels(self):
        '''Returns all of the channels parsed from the file as a 2D numpy array'''

        time_list = [f[1] for f in self.values]
        return np.asarray([[channel[2] for channel in time] for time in time_list])

    def get_skeleton_tree(self):
        tree = []
        root_key = [j for j in self.skeleton if self.skeleton[j]['parent'] == None][0]

        root_joint = Joint(root_key)

    def get_base_pose(self):
        return self.base_pose

    def get_empty_channels(self):
        # TODO
        pass

    def get_constant_channels(self):
        # TODO
        pass


class ASF_AMC():
    def __init__(self):
        self.skeleton = {}
        self.values = None
        self.channel_names = []
        self.root_name = ''

    def traverse(self, j=None):
        stack = [self.root_name]
        while stack:
            joint = stack.pop()
            yield joint
            for c in self.skeleton[joint]['children']:
                stack.append(c)

    def clone(self):
        import copy
        new_data = ASF_AMC()
        new_data.skeleton = copy.deepcopy(self.skeleton)
        new_data.values = copy.deepcopy(self.values)
        new_data.channel_names = copy.deepcopy(self.channel_names)
        new_data.root_name = copy.deepcopy(self.root_name)
        return new_data

    def get_all_channels(self):
        '''Returns all of the channels parsed from the file as a 2D numpy array'''

        time_list = [f[1] for f in self.values]
        return np.asarray([[channel[2] for channel in time] for time in time_list])

    def get_skeleton_tree(self):
        tree = []
        root_key = [j for j in self.skeleton if self.skeleton[j]['parent'] == None][0]

        root_joint = Joint(root_key)

    def get_base_pose(self):
        return self.base_pose

    def get_empty_channels(self):
        # TODO
        pass

    def get_constant_channels(self):
        # TODO
        pass