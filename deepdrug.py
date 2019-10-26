# -*- coding: utf-8  -*-

"""Module qui contient tous les fonctions utilisés pour le projet : Deep Learning Binding pocket
"""
import sys
import os
import argparse

import numpy as np
import pandas as pd
import glob

from keras import callbacks
from keras.optimizers import Adam
from keras.utils import np_utils


#Lecture des datas :
def load_ligand(file):
    ''' read a ligand file and return a np data frame
    '''
    list_ligand = np.loadtxt(file, dtype = str)
    return list_ligand

def load_voxel(voxel_folder , nucleotide_list , hemes_list,
    steroid_list, control_list):
    ''' take a folder , a list of ligand and return a
    list of targets
    '''
    voxel_files = glob.glob(voxel_folder+'\*.npy')
    #The glob.glob returns the list of files with their
    #full path (unlike os.listdir())
    random.shuffle(voxel_files)
    targets= []
    features= []
    controls= []
    steroids = []

    for voxel in voxel_files[:200]:
        name = voxel[22:,-4]
        if name in nucleotide_list:
            features.append(np.load(voxel))
            target = [1,0]
            targets.append(target)
        elif name in hemes_list:
            features.append(np.load(voxel))
            target = [0,1]
            targets.append(target)
        elif name in steroid_list:
            steriod = np.load(voxel)
            steroides.append(np.reshape(steroid,(14,32,32,32)))
        elif name in control:
            c = np.load(voxel)
            controls.append(np.reshape(c,(14,32,32,32)))
        else:
            break

    features = np.array(features)
    targets = np.array(targets)
    controls = np.array(controls)
    steroides = np.array(steroides)

    np.save('features', features)
    np.save('targets', targets)
    np.save('controls', controls)
    np.save('steroides', steroides)
