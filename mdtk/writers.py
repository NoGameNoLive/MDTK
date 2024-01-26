'''
-*- coding: utf-8 -*-
time: 2023/4/15 13:10
file: writers.py
author: Endy_Liu_Noonell
'''

import numpy as np
import pandas as pd


class BVHWriter():
    def __init__(self):
        pass

    def write(self, X, ofile, framerate=-1, start=0, stop=-1):

        # 编写skeleton信息
        ofile.write('HIERARCHY\n')

        self.motions_ = []
        self._printJoint(X, X.root_name, 0, ofile)

        if stop > 0:
            nframes = stop - start
        else:
            nframes = X.values.shape[0]
            stop = X.values.shape[0]

        # Writing the motion header
        ofile.write('MOTION\n')
        ofile.write('Frames: %d\n' % nframes)
        # print(framerate)

        ofile.write('Frame Time: %f\n' % framerate)
        # if framerate > 0:
        #     ofile.write('Frame Time: %f\n' % float(1.0 / framerate))
        # else:
        #     ofile.write('Frame Time: %f\n' % framerate)

        # Writing the data
        self.motions_ = np.asarray(self.motions_).T
        lines = [" ".join(item) for item in self.motions_[start:stop].astype(str)]
        ofile.write("".join("%s\n" % l for l in lines))

    def _printJoint(self, X, joint, tab, ofile):

        if X.skeleton[joint]['parent'] == None:
            ofile.write('ROOT %s\n' % joint)
        elif len(X.skeleton[joint]['children']) > 0:
            ofile.write('%sJOINT %s\n' % ('\t' * (tab), joint))
        else:
            ofile.write('%sEnd site\n' % ('\t' * (tab)))

        ofile.write('%s{\n' % ('\t' * (tab)))

        ofile.write('%sOFFSET %3.5f %3.5f %3.5f\n' % ('\t' * (tab + 1),
                                                      X.skeleton[joint]['offsets'][0],
                                                      X.skeleton[joint]['offsets'][1],
                                                      X.skeleton[joint]['offsets'][2]))
        rot_order = X.skeleton[joint]['order']

        # print("rot_order = " + rot_order)
        channels = X.skeleton[joint]['channels']
        rot = [c for c in channels if ('rotation' in c)]
        pos = [c for c in channels if ('position' in c)]

        n_channels = len(rot) + len(pos)
        ch_str = ''
        if n_channels > 0:
            for ci in range(len(pos)):
                cn = pos[ci]
                self.motions_.append(np.asarray(X.values['%s_%s' % (joint, cn)].values))
                ch_str = ch_str + ' ' + cn
            for ci in range(len(rot)):
                cn = '%srotation' % (rot_order[ci])
                self.motions_.append(np.asarray(X.values['%s_%s' % (joint, cn)].values))
                ch_str = ch_str + ' ' + cn
        if len(X.skeleton[joint]['children']) > 0:
            # ch_str = ''.join(' %s'*n_channels%tuple(channels))
            ofile.write('%sCHANNELS %d%s\n' % ('\t' * (tab + 1), n_channels, ch_str))

            for c in X.skeleton[joint]['children']:
                self._printJoint(X, c, tab + 1, ofile)

        ofile.write('%s}\n' % ('\t' * (tab)))


class PPMWriter():
    def __init__(self):
        pass

    def write(self, X, ofile, start=0, stop=-1):

        # 编写skeleton信息
        ofile.write('HIERARCHY\n')

        self.motions_ = []
        self._printJoint(X, X.root_name, 0, ofile)

        base_pos = X.base_pose
        time_list = X.time_list

        if stop > 0:
            nframes = stop - start
        else:
            nframes = X.values.shape[0]
            stop = X.values.shape[0]

        # Writing the motion header
        ofile.write('BasePose\n')
        for item in base_pos:
            ofile.write('%3.5f ' % base_pos[item])
            # ofile.write('%3.5f ' % item)
        ofile.write('\nTime: ')
        ofile.write('%d\n' % len(time_list))
        for time in time_list:
            ofile.write('%3.5f ' % time)
        ofile.write('\nMOTION\n')
        # ofile.write('Frames: %d\n' % nframes)
        # # print(framerate)
        #
        # ofile.write('Frame Time: %f\n' % framerate)
        # if framerate > 0:
        #     ofile.write('Frame Time: %f\n' % float(1.0 / framerate))
        # else:
        #     ofile.write('Frame Time: %f\n' % framerate)

        # Writing the data
        self.motions_ = np.asarray(self.motions_).T
        lines = [" ".join(item) for item in self.motions_[start:stop].astype(str)]
        ofile.write("".join("%s\n" % l for l in lines))

    def _printJoint(self, X, joint, tab, ofile):

        if X.skeleton[joint]['parent'] == None:
            ofile.write('ROOT %s\n' % joint)
        elif len(X.skeleton[joint]['children']) >= 0:
            ofile.write('%sJOINT %s\n' % ('\t' * (tab), joint))
        # else:
        #     ofile.write('%sEnd site\n' % ('\t' * (tab)))
        #     return

        ofile.write('%s{\n' % ('\t' * (tab)))

        ofile.write('%sLENGTH %3.5f\n' % ('\t' * (tab + 1), X.skeleton[joint]['length']))

        rot_order = X.skeleton[joint]['order']

        # print("rot_order = " + rot_order)
        channels = X.skeleton[joint]['channels']
        acc = [c for c in channels if ('accelerate' in c)]

        n_channels = len(acc)
        ch_str = ''
        if n_channels > 0:
            for ci in range(len(acc)):
                cn = '%saccelerate' % (rot_order[ci])
                # cn = '%saccelerate' % (rot_order[ci])
                self.motions_.append(np.asarray(X.values['%s_%s' % (joint, cn)].values))
                ch_str = ch_str + ' ' + cn
        if len(X.skeleton[joint]['children']) >= 0:
            # ch_str = ''.join(' %s'*n_channels%tuple(channels))
            ofile.write('%sCHANNELS %d%s\n' % ('\t' * (tab + 1), n_channels, ch_str))

            for c in X.skeleton[joint]['children']:
                self._printJoint(X, c, tab + 1, ofile)
        if len(X.skeleton[joint]['children']) == 0:
            ofile.write('%sEnd site\n' % ('\t' * (tab + 1)))
            # return

        ofile.write('%s}\n' % ('\t' * (tab)))


class CSMDWriter():
    def __init__(self):
        pass

    def write(self, X, ofile, start=0, stop=-1):

        # write skeleton
        ofile.write('SKELETON\n')

        self.motions_ = []
        self._printJoint(X, X.root_name, 0, ofile)

        time_list = X.time_list

        if stop > 0:
            nframes = stop - start
        else:
            nframes = X.values.shape[0]
            stop = X.values.shape[0]

        # write basic
        ofile.write('BASIC\n')
        ofile.write('Time: ')
        ofile.write('%d\n' % len(time_list))
        for time in time_list:
            ofile.write('%3.5f ' % time)
        ofile.write('\nPOSITION: \n')
        for position in X.root_position:
            for pos in position:
                ofile.write('%3.5f ' % pos)
        ofile.write('\nDIRECTION: \n')
        for direction in X.root_direction:
            # print(direction)
            for dir_line in direction.tolist():
                # print(dir_line)
                for dir in dir_line:
                    # print(dir)
                    ofile.write('%3.5f ' % dir)

        ofile.write('\nMOTION\n')

        # Writing the data
        self.motions_ = np.asarray(self.motions_).T
        lines = [" ".join(item) for item in self.motions_[start:stop].astype(str)]
        ofile.write("".join("%s\n" % l for l in lines))

    def _printJoint(self, X, joint, tab, ofile):

        if X.skeleton[joint]['parent'] == None:
            ofile.write('ROOT %s\n' % joint)
        elif len(X.skeleton[joint]['children']) == 0:
            ofile.write('%sENDSITE %s\n' % ('\t' * (tab), joint))
        elif len(X.skeleton[joint]['children']) > 0:
            ofile.write('%sJOINT %s\n' % ('\t' * (tab), joint))
        # else:
        #     ofile.write('%sEnd site\n' % ('\t' * (tab)))
        #     return

        ofile.write('%s{\n' % ('\t' * (tab)))

        ofile.write('%sLENGTH %3.5f\n' % ('\t' * (tab + 1), X.skeleton[joint]['length']))

        ofile.write('%sTYPE %s\n' % ('\t' * (tab + 1), X.skeleton[joint]['type']))

        # print("rot_order = " + rot_order)
        channels = X.skeleton[joint]['channels']

        n_channels = len(channels)
        ch_str = ''
        if n_channels > 0:
            for ci in range(n_channels):
                cn = channels[ci]
                self.motions_.append(np.asarray(X.values['%s_%s' % (joint, cn)].values))
                ch_str = ch_str + ' ' + cn
        if len(X.skeleton[joint]['children']) >= 0:
            # ch_str = ''.join(' %s'*n_channels%tuple(channels))
            ofile.write('%sCHANNELS %d%s\n' % ('\t' * (tab + 1), n_channels, ch_str))
            if len(X.skeleton[joint]['children']) > 0:
                for c in X.skeleton[joint]['children']:
                    self._printJoint(X, c, tab + 1, ofile)

        ofile.write('%s}\n' % ('\t' * (tab)))
