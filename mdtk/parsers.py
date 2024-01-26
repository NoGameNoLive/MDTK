'''
-*- coding: utf-8 -*-
time: 2023/4/15 13:05
file: parsers.py
author: Endy_Liu_Noonell
'''

import re
import numpy as np
from .data import Joint, MocapData, CSMD


class BVHScanner():
    '''
    A wrapper class for re.Scanner
    '''

    def __init__(self):
        def identifier(scanner, token):
            return 'IDENT', token

        def operator(scanner, token):
            return 'OPERATOR', token

        def digit(scanner, token):
            return 'DIGIT', token

        def open_brace(scanner, token):
            return 'OPEN_BRACE', token

        def close_brace(scanner, token):
            return 'CLOSE_BRACE', token

        self.scanner = re.Scanner([
            (r'[a-zA-Z_]\w*', identifier),
            # (r'-*[0-9]+(\.[0-9]+)?', digit), # won't work for .34
            # (r'[-+]?[0-9]*\.?[0-9]+', digit), # won't work for 4.56e-2
            # (r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', digit),
            (r'-*[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', digit),
            (r'}', close_brace),
            (r'}', close_brace),
            (r'{', open_brace),
            (r':', None),
            (r'\s+', None)
        ])

    def scan(self, stuff):
        return self.scanner.scan(stuff)


class CSMDScanner():
    '''
    A wrapper class for re.Scanner
    '''

    def __init__(self):
        def identifier(scanner, token):
            return 'IDENT', token

        def operator(scanner, token):
            return 'OPERATOR', token

        def digit(scanner, token):
            return 'DIGIT', token

        def open_brace(scanner, token):
            return 'OPEN_BRACE', token

        def close_brace(scanner, token):
            return 'CLOSE_BRACE', token

        self.scanner = re.Scanner([
            (r'[a-zA-Z_]\w*', identifier),
            # (r'-*[0-9]+(\.[0-9]+)?', digit), # won't work for .34
            # (r'[-+]?[0-9]*\.?[0-9]+', digit), # won't work for 4.56e-2
            # (r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', digit),
            (r'-*[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', digit),
            (r'}', close_brace),
            (r'}', close_brace),
            (r'{', open_brace),
            (r':', None),
            (r'\s+', None)
        ])

    def scan(self, stuff):
        return self.scanner.scan(stuff)


class ASF_AMCScanner():
    '''
    A wrapper class for re.Scanner
    '''

    def __init__(self):
        def identifier(scanner, token):
            return 'IDENT', token

        def operator(scanner, token):
            return 'OPERATOR', token

        def digit(scanner, token):
            return 'DIGIT', token

        def open_brace(scanner, token):
            return 'OPEN_BRACE', token

        def close_brace(scanner, token):
            return 'CLOSE_BRACE', token

        self.scanner = re.Scanner([
            (r'[a-zA-Z_]\w*', identifier),
            # (r'-*[0-9]+(\.[0-9]+)?', digit), # won't work for .34
            # (r'[-+]?[0-9]*\.?[0-9]+', digit), # won't work for 4.56e-2
            # (r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', digit),
            (r'-*[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', digit),
            (r'}', close_brace),
            (r'}', close_brace),
            (r'{', open_brace),
            (r':', None),
            (r'\s+', None)
        ])

    def scan(self, stuff):
        return self.scanner.scan(stuff)


class BVHParser():
    '''
    A class to parse a PPM file.

    Extracts the skeleton and channel values
    '''

    def __init__(self, filename=None):
        self.reset()

    def reset(self):
        self._skeleton = {}
        self.bone_context = []
        self._motion_channels = []
        self._motions = []
        self.current_token = 0
        self.framerate = 0.0
        self.root_name = ''

        self.scanner = BVHScanner()

        self.data = MocapData()

    def parse(self, filename, start=0, stop=-1):
        self.reset()

        with open(filename, 'r') as bvh_file:
            raw_contents = bvh_file.read()
        tokens, remainder = self.scanner.scan(raw_contents)
        self._parse_hierarchy(tokens)
        self.current_token = self.current_token + 1
        self._parse_motion(tokens, start, stop)

        self.data.skeleton = self._skeleton
        self.data.channel_names = self._motion_channels
        self.data.values = self._to_DataFrame()
        self.data.root_name = self.root_name
        self.data.framerate = self.framerate

        return self.data

    def _to_DataFrame(self):
        '''Returns all of the channels parsed from the file as a pandas DataFrame'''

        import pandas as pd
        # time_index = pd.to_timedelta([f[0] for f in self._motions], unit='s')
        time_index = [f[0] for f in self._motions]
        frames = [f[1] for f in self._motions]
        channels = np.asarray([[channel[2] for channel in frame] for frame in frames])
        column_names = ['%s_%s' % (c[0], c[1]) for c in self._motion_channels]

        return pd.DataFrame(data=channels, index=time_index, columns=column_names)

    def _new_bone(self, parent, name):
        bone = {'parent': parent, 'channels': [], 'offsets': [], 'order': '', 'children': []}
        return bone

    def _push_bone_context(self, name):
        self.bone_context.append(name)

    def _get_bone_context(self):
        return self.bone_context[len(self.bone_context) - 1]

    def _pop_bone_context(self):
        self.bone_context = self.bone_context[:-1]
        return self.bone_context[len(self.bone_context) - 1]

    def _read_offset(self, bvh, token_index):
        if bvh[token_index] != ('IDENT', 'OFFSET'):
            return None, None
        token_index = token_index + 1
        offsets = [0.0] * 3
        for i in range(3):
            offsets[i] = float(bvh[token_index][1])
            token_index = token_index + 1
        return offsets, token_index

    def _read_channels(self, bvh, token_index):
        if bvh[token_index] != ('IDENT', 'CHANNELS'):
            return None, None
        token_index = token_index + 1
        channel_count = int(bvh[token_index][1])
        token_index = token_index + 1
        channels = [""] * channel_count
        order = ""
        for i in range(channel_count):
            channels[i] = bvh[token_index][1]
            token_index = token_index + 1
            if (channels[i] == "Xrotation" or channels[i] == "Yrotation" or channels[i] == "Zrotation"):
                order += channels[i][0]
            else:
                order = ""
        return channels, token_index, order

    def _parse_joint(self, bvh, token_index):
        end_site = False
        joint_id = bvh[token_index][1]
        token_index = token_index + 1
        joint_name = bvh[token_index][1]
        token_index = token_index + 1

        parent_name = self._get_bone_context()

        if (joint_id == "End"):
            joint_name = parent_name + '_Nub'
            end_site = True
        joint = self._new_bone(parent_name, joint_name)
        if bvh[token_index][0] != 'OPEN_BRACE':
            print('Was expecting brance, got ', bvh[token_index])
            return None
        token_index = token_index + 1
        offsets, token_index = self._read_offset(bvh, token_index)
        joint['offsets'] = offsets
        if not end_site:
            channels, token_index, order = self._read_channels(bvh, token_index)
            joint['channels'] = channels
            joint['order'] = order
            for channel in channels:
                self._motion_channels.append((joint_name, channel))

        self._skeleton[joint_name] = joint
        self._skeleton[parent_name]['children'].append(joint_name)

        while (bvh[token_index][0] == 'IDENT' and bvh[token_index][1] == 'JOINT') or (
                bvh[token_index][0] == 'IDENT' and bvh[token_index][1] == 'End'):
            self._push_bone_context(joint_name)
            token_index = self._parse_joint(bvh, token_index)
            self._pop_bone_context()

        if bvh[token_index][0] == 'CLOSE_BRACE':
            return token_index + 1

        print('Unexpected token ', bvh[token_index])

    def _parse_hierarchy(self, bvh):
        self.current_token = 0
        if bvh[self.current_token] != ('IDENT', 'HIERARCHY'):
            return None
        self.current_token = self.current_token + 1
        if bvh[self.current_token] != ('IDENT', 'ROOT'):
            return None
        self.current_token = self.current_token + 1
        if bvh[self.current_token][0] != 'IDENT':
            return None

        root_name = bvh[self.current_token][1]
        root_bone = self._new_bone(None, root_name)
        self.current_token = self.current_token + 2  # skipping open brace
        offsets, self.current_token = self._read_offset(bvh, self.current_token)
        channels, self.current_token, order = self._read_channels(bvh, self.current_token)
        root_bone['offsets'] = offsets
        root_bone['channels'] = channels
        root_bone['order'] = order
        self._skeleton[root_name] = root_bone
        self._push_bone_context(root_name)

        for channel in channels:
            self._motion_channels.append((root_name, channel))

        while bvh[self.current_token][1] == 'JOINT':
            self.current_token = self._parse_joint(bvh, self.current_token)

        self.root_name = root_name

    def _parse_motion(self, bvh, start, stop):
        if bvh[self.current_token][0] != 'IDENT':
            print('Unexpected text')
            return None
        if bvh[self.current_token][1] != 'MOTION':
            print('No motion section')
            return None
        self.current_token = self.current_token + 1
        if bvh[self.current_token][1] != 'Frames':
            return None
        self.current_token = self.current_token + 1
        frame_count = int(bvh[self.current_token][1])

        if stop < 0 or stop > frame_count:
            stop = frame_count

        assert (start >= 0)
        assert (start < stop)

        self.current_token = self.current_token + 1
        if bvh[self.current_token][1] != 'Frame':
            return None
        self.current_token = self.current_token + 1
        if bvh[self.current_token][1] != 'Time':
            return None
        self.current_token = self.current_token + 1
        frame_rate = float(bvh[self.current_token][1])

        self.framerate = frame_rate

        self.current_token = self.current_token + 1

        frame_time = 0.0
        self._motions = [()] * (stop - start)
        idx = 0
        for i in range(stop):
            channel_values = []
            for channel in self._motion_channels:
                # print(channel)
                channel_values.append((channel[0], channel[1], float(bvh[self.current_token][1])))
                self.current_token = self.current_token + 1

            if i >= start:
                self._motions[idx] = (frame_time, channel_values)
                frame_time = frame_time + frame_rate
                idx += 1


class CSMDParser():
    '''
    A class to parse a CSMD file.

    Extracts the skeleton and channel values
    '''

    def __init__(self, filename=None):
        self.reset()

    def reset(self):
        self._skeleton = {}
        self.bone_context = []
        self._motion_channels = []
        self._motions = []
        self.current_token = 0
        self.time_list = []
        self.root_name = ''
        self.root_position = []
        self.root_direction = []
        self.endsites = []

        self.scanner = CSMDScanner()

        self.data = CSMD()

    def parse(self, filename, start=0, stop=-1):
        self.reset()

        with open(filename, 'r') as csmd_file:
            raw_contents = csmd_file.read()
        tokens, remainder = self.scanner.scan(raw_contents)
        self._parse_hierarchy(tokens)
        self._parse_basic(tokens)
        # print(tokens[self.current_token])
        # self.current_token = self.current_token + 1
        self._parse_motion(tokens, start, stop)

        self.data.skeleton = self._skeleton
        self.data.channel_names = self._motion_channels
        self.data.values = self._to_DataFrame()
        self.data.root_name = self.root_name
        self.data.time_list = self.time_list
        self.data.endsite_list = self.endsites
        self.data.root_position = self.root_position
        self.data.root_direction = self.root_direction

        return self.data

    def _to_DataFrame(self):
        '''Returns all of the channels parsed from the file as a pandas DataFrame'''

        import pandas as pd
        # time_index = pd.to_timedelta([f[0] for f in self._motions], unit='s')
        time_index = self.time_list
        frames = [f for f in self._motions]
        channels = np.asarray([[channel[2] for channel in frame] for frame in frames])
        column_names = ['%s_%s' % (c[0], c[1]) for c in self._motion_channels]

        return pd.DataFrame(data=channels, index=time_index, columns=column_names)

    def _new_node(self, parent, name):
        bone = {'parent': parent, 'channels': [], 'type': '', 'length': [], 'order': '', 'children': []}
        return bone

    def _push_bone_context(self, name):
        self.bone_context.append(name)

    def _get_bone_context(self):
        return self.bone_context[len(self.bone_context) - 1]

    def _pop_bone_context(self):
        self.bone_context = self.bone_context[:-1]
        return self.bone_context[len(self.bone_context) - 1]

    def _read_length(self, csmd, token_index):
        if csmd[token_index] != ('IDENT', 'LENGTH'):
            return None, None
        token_index = token_index + 1
        length = float(csmd[token_index][1])
        token_index = token_index + 1
        return length, token_index

    def _read_type(self, csmd, token_index):
        if csmd[token_index] != ('IDENT', 'TYPE'):
            return None, None
        token_index = token_index + 1
        node_type = csmd[token_index][1]
        token_index = token_index + 1
        return node_type, token_index

    def _read_channels(self, csmd, token_index):
        if csmd[token_index] != ('IDENT', 'CHANNELS'):
            return None, None
        token_index = token_index + 1
        channel_count = int(csmd[token_index][1])
        token_index = token_index + 1
        channels = [""] * channel_count
        order = ""
        for i in range(channel_count):
            channels[i] = csmd[token_index][1]
            token_index = token_index + 1
            if (channels[i] == "Xvector" or channels[i] == "Yvector" or channels[i] == "Zvector" or channels[
                i] == "Arotation"):
                order += channels[i][0]
            else:
                order = ""
        return channels, token_index, order

    def _parse_joint(self, csmd, token_index):
        node_id = csmd[token_index][1]
        token_index = token_index + 1
        joint_name = csmd[token_index][1]
        token_index = token_index + 1

        parent_name = self._get_bone_context()

        joint = self._new_node(parent_name, joint_name)
        if csmd[token_index][0] != 'OPEN_BRACE':
            print('Was expecting brance, got ', csmd[token_index])
            return None
        token_index = token_index + 1
        length, token_index = self._read_length(csmd, token_index)
        joint['length'] = length
        node_type, token_index = self._read_type(csmd, token_index)
        joint['type'] = node_type
        channels, token_index, order = self._read_channels(csmd, token_index)
        joint['channels'] = channels
        joint['order'] = order
        for channel in channels:
            self._motion_channels.append((joint_name, channel))

        self._skeleton[joint_name] = joint
        self._skeleton[parent_name]['children'].append(joint_name)
        if node_id == 'ENDSITE':
            self.endsites.append(joint_name)

        # print(token_index)
        while (csmd[token_index][0] == 'IDENT' and node_id == 'JOINT'):
            self._push_bone_context(joint_name)
            token_index = self._parse_joint(csmd, token_index)
            self._pop_bone_context()
            # if csmd[token_index][1] == 'JOINT':
            #     self._push_bone_context(joint_name)
            #     token_index = self._parse_joint(csmd, token_index)
            #     self._pop_bone_context()
            # if csmd[token_index][1] == 'ENDSITE':
            #     token_index = self._parse_joint(csmd, token_index)

        if csmd[token_index][0] == 'CLOSE_BRACE':
            return token_index + 1

        print('Unexpected token ', csmd[token_index])

    def _parse_hierarchy(self, csmd):
        self.current_token = 0
        if csmd[self.current_token] != ('IDENT', 'SKELETON'):
            return None
        self.current_token = self.current_token + 1
        if csmd[self.current_token] != ('IDENT', 'ROOT'):
            return None
        self.current_token = self.current_token + 1
        if csmd[self.current_token][0] != 'IDENT':
            return None

        root_name = csmd[self.current_token][1]
        root_node = self._new_node(None, root_name)
        self.current_token = self.current_token + 2  # skipping open brace
        length, self.current_token = self._read_length(csmd, self.current_token)
        node_type, self.current_token = self._read_type(csmd, self.current_token)
        # print(csmd[self.current_token][1])
        channels, self.current_token, order = self._read_channels(csmd, self.current_token)
        root_node['channels'] = channels
        root_node['type'] = node_type
        root_node['length'] = length
        self._skeleton[root_name] = root_node
        self._push_bone_context(root_name)

        for channel in channels:
            self._motion_channels.append((root_name, channel))

        while csmd[self.current_token][1] == 'JOINT':
            self.current_token = self._parse_joint(csmd, self.current_token)
            while csmd[self.current_token][0] == 'CLOSE_BRACE':
                self.current_token = self.current_token + 1
            # print(csmd[self.current_token][1])

        self.root_name = root_name

    def _parse_basic(self, csmd):
        if csmd[self.current_token] != ('IDENT', 'BASIC'):
            return None
        self.current_token = self.current_token + 1

        if csmd[self.current_token][1] != 'Time':
            print('No Time')
            return None
        self.current_token = self.current_token + 1
        time_num = int(csmd[self.current_token][1])
        self.current_token = self.current_token + 1
        for i in range(time_num):
            self.time_list.append(csmd[self.current_token][1])
            self.current_token = self.current_token + 1

        if csmd[self.current_token][1] != 'POSITION':
            print('No Root Postion')
            return None
        self.current_token = self.current_token + 1
        for time in range(time_num):
            position_clip = []
            for i in range(3):
                position_clip.append(float(csmd[self.current_token][1]))
                self.current_token = self.current_token + 1
            self.root_position.append(position_clip)

        if csmd[self.current_token][1] != 'DIRECTION':
            print('No Root Direction')
            return None
        self.current_token = self.current_token + 1
        for time in range(time_num):
            direction_clip = []
            for i in range(3):
                dir_m = []
                for j in range(3):
                    dir_m.append(float(csmd[self.current_token][1]))
                    self.current_token = self.current_token + 1
                direction_clip.append(dir_m)
            self.root_direction.append(direction_clip)

    def _parse_motion(self, csmd, start, stop):
        if csmd[self.current_token][0] != 'IDENT':
            print('Unexpected text')
            return None

        if csmd[self.current_token][1] != 'MOTION':
            print('No motion section')
            return None
        self.current_token = self.current_token + 1

        if stop < 0 or stop > len(self.time_list):
            stop = len(self.time_list)

        assert (start >= 0)
        assert (start < stop)

        self._motions = [()] * (stop - start)
        idx = 0
        for i in range(stop):
            channel_values = []
            for channel in self._motion_channels:
                # print(channel)
                channel_values.append((channel[0], channel[1], float(csmd[self.current_token][1])))
                self.current_token = self.current_token + 1

            if i >= start:
                self._motions[idx] = channel_values
                idx += 1

class ASF_AMCParser():
    '''
    A class to parse a ASF_AMC file.

    Extracts the skeleton and channel values
    '''

    def __init__(self, filename=None):
        self.reset()

    def reset(self):
        self._skeleton = {}
        self.bone_context = []
        self._motion_channels = []
        self._motions = []
        self.current_token = 0
        self.time_list = []
        self.root_name = ''
        self.root_position = []
        self.root_direction = []
        self.endsites = []

        self.scanner = CSMDScanner()

        self.data = CSMD()

    def parse(self, filename, start=0, stop=-1):
        self.reset()

        with open(filename, 'r') as csmd_file:
            raw_contents = csmd_file.read()
        tokens, remainder = self.scanner.scan(raw_contents)
        self._parse_hierarchy(tokens)
        self._parse_basic(tokens)
        # print(tokens[self.current_token])
        # self.current_token = self.current_token + 1
        self._parse_motion(tokens, start, stop)

        self.data.skeleton = self._skeleton
        self.data.channel_names = self._motion_channels
        self.data.values = self._to_DataFrame()
        self.data.root_name = self.root_name
        self.data.time_list = self.time_list
        self.data.endsite_list = self.endsites
        self.data.root_position = self.root_position
        self.data.root_direction = self.root_direction

        return self.data

    def _to_DataFrame(self):
        '''Returns all of the channels parsed from the file as a pandas DataFrame'''

        import pandas as pd
        # time_index = pd.to_timedelta([f[0] for f in self._motions], unit='s')
        time_index = self.time_list
        frames = [f for f in self._motions]
        channels = np.asarray([[channel[2] for channel in frame] for frame in frames])
        column_names = ['%s_%s' % (c[0], c[1]) for c in self._motion_channels]

        return pd.DataFrame(data=channels, index=time_index, columns=column_names)

    def _new_node(self, parent, name):
        bone = {'parent': parent, 'channels': [], 'type': '', 'length': [], 'order': '', 'children': []}
        return bone

    def _push_bone_context(self, name):
        self.bone_context.append(name)

    def _get_bone_context(self):
        return self.bone_context[len(self.bone_context) - 1]

    def _pop_bone_context(self):
        self.bone_context = self.bone_context[:-1]
        return self.bone_context[len(self.bone_context) - 1]

    def _read_length(self, csmd, token_index):
        if csmd[token_index] != ('IDENT', 'LENGTH'):
            return None, None
        token_index = token_index + 1
        length = float(csmd[token_index][1])
        token_index = token_index + 1
        return length, token_index

    def _read_type(self, csmd, token_index):
        if csmd[token_index] != ('IDENT', 'TYPE'):
            return None, None
        token_index = token_index + 1
        node_type = csmd[token_index][1]
        token_index = token_index + 1
        return node_type, token_index

    def _read_channels(self, csmd, token_index):
        if csmd[token_index] != ('IDENT', 'CHANNELS'):
            return None, None
        token_index = token_index + 1
        channel_count = int(csmd[token_index][1])
        token_index = token_index + 1
        channels = [""] * channel_count
        order = ""
        for i in range(channel_count):
            channels[i] = csmd[token_index][1]
            token_index = token_index + 1
            if (channels[i] == "Xvector" or channels[i] == "Yvector" or channels[i] == "Zvector" or channels[
                i] == "Arotation"):
                order += channels[i][0]
            else:
                order = ""
        return channels, token_index, order

    def _parse_joint(self, csmd, token_index):
        node_id = csmd[token_index][1]
        token_index = token_index + 1
        joint_name = csmd[token_index][1]
        token_index = token_index + 1

        parent_name = self._get_bone_context()

        joint = self._new_node(parent_name, joint_name)
        if csmd[token_index][0] != 'OPEN_BRACE':
            print('Was expecting brance, got ', csmd[token_index])
            return None
        token_index = token_index + 1
        length, token_index = self._read_length(csmd, token_index)
        joint['length'] = length
        node_type, token_index = self._read_type(csmd, token_index)
        joint['type'] = node_type
        channels, token_index, order = self._read_channels(csmd, token_index)
        joint['channels'] = channels
        joint['order'] = order
        for channel in channels:
            self._motion_channels.append((joint_name, channel))

        self._skeleton[joint_name] = joint
        self._skeleton[parent_name]['children'].append(joint_name)
        if node_id == 'ENDSITE':
            self.endsites.append(joint_name)

        # print(token_index)
        while (csmd[token_index][0] == 'IDENT' and node_id == 'JOINT'):
            self._push_bone_context(joint_name)
            token_index = self._parse_joint(csmd, token_index)
            self._pop_bone_context()
            # if csmd[token_index][1] == 'JOINT':
            #     self._push_bone_context(joint_name)
            #     token_index = self._parse_joint(csmd, token_index)
            #     self._pop_bone_context()
            # if csmd[token_index][1] == 'ENDSITE':
            #     token_index = self._parse_joint(csmd, token_index)

        if csmd[token_index][0] == 'CLOSE_BRACE':
            return token_index + 1

        print('Unexpected token ', csmd[token_index])

    def _parse_hierarchy(self, csmd):
        self.current_token = 0
        if csmd[self.current_token] != ('IDENT', 'SKELETON'):
            return None
        self.current_token = self.current_token + 1
        if csmd[self.current_token] != ('IDENT', 'ROOT'):
            return None
        self.current_token = self.current_token + 1
        if csmd[self.current_token][0] != 'IDENT':
            return None

        root_name = csmd[self.current_token][1]
        root_node = self._new_node(None, root_name)
        self.current_token = self.current_token + 2  # skipping open brace
        length, self.current_token = self._read_length(csmd, self.current_token)
        node_type, self.current_token = self._read_type(csmd, self.current_token)
        # print(csmd[self.current_token][1])
        channels, self.current_token, order = self._read_channels(csmd, self.current_token)
        root_node['channels'] = channels
        root_node['type'] = node_type
        root_node['length'] = length
        self._skeleton[root_name] = root_node
        self._push_bone_context(root_name)

        for channel in channels:
            self._motion_channels.append((root_name, channel))

        while csmd[self.current_token][1] == 'JOINT':
            self.current_token = self._parse_joint(csmd, self.current_token)
            while csmd[self.current_token][0] == 'CLOSE_BRACE':
                self.current_token = self.current_token + 1
            # print(csmd[self.current_token][1])

        self.root_name = root_name

    def _parse_basic(self, csmd):
        if csmd[self.current_token] != ('IDENT', 'BASIC'):
            return None
        self.current_token = self.current_token + 1

        if csmd[self.current_token][1] != 'Time':
            print('No Time')
            return None
        self.current_token = self.current_token + 1
        time_num = int(csmd[self.current_token][1])
        self.current_token = self.current_token + 1
        for i in range(time_num):
            self.time_list.append(csmd[self.current_token][1])
            self.current_token = self.current_token + 1

        if csmd[self.current_token][1] != 'POSITION':
            print('No Root Postion')
            return None
        self.current_token = self.current_token + 1
        for time in range(time_num):
            position_clip = []
            for i in range(3):
                position_clip.append(float(csmd[self.current_token][1]))
                self.current_token = self.current_token + 1
            self.root_position.append(position_clip)

        if csmd[self.current_token][1] != 'DIRECTION':
            print('No Root Direction')
            return None
        self.current_token = self.current_token + 1
        for time in range(time_num):
            direction_clip = []
            for i in range(3):
                dir_m = []
                for j in range(3):
                    dir_m.append(float(csmd[self.current_token][1]))
                    self.current_token = self.current_token + 1
                direction_clip.append(dir_m)
            self.root_direction.append(direction_clip)

    def _parse_motion(self, csmd, start, stop):
        if csmd[self.current_token][0] != 'IDENT':
            print('Unexpected text')
            return None

        if csmd[self.current_token][1] != 'MOTION':
            print('No motion section')
            return None
        self.current_token = self.current_token + 1

        if stop < 0 or stop > len(self.time_list):
            stop = len(self.time_list)

        assert (start >= 0)
        assert (start < stop)

        self._motions = [()] * (stop - start)
        idx = 0
        for i in range(stop):
            channel_values = []
            for channel in self._motion_channels:
                # print(channel)
                channel_values.append((channel[0], channel[1], float(csmd[self.current_token][1])))
                self.current_token = self.current_token + 1

            if i >= start:
                self._motions[idx] = channel_values
                idx += 1
