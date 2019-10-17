# -*- coding: utf-8  -*-

"""Module qui contient tous les fonctions utilisés pour le projet : Deep Learning Binding pocket
"""

import sys
import os
import argparse

import numpy as np

#from deepdrug3d import DeepDrug3DBuilder

from keras import callbacks
from keras.optimizers import Adam
from keras.utils import np_utils


#Lecture des datas :
atps = []
with open('data/nucleotide.list.txt') as atp_in:
    for line in atp_in.readlines():
            temp = line.replace(' ','').replace('\n','')
            atps.append(temp)

hemes = []
with open('data/heme.list.txt') as heme_in:
    for line in heme_in.readlines():
         temp = line.replace(' ','').replace('\n','')
         hemes.append(temp)

atp_len = len(atps)
heme_len = len(hemes)
L = atp_len + heme_len
voxel = np.zeros(shape = (L, 14, 32, 32, 32),dtype = np.float64)
label = np.zeros(shape = (L,), dtype = int)
cnt = 0

#Lecture des fichiers .npy
'''for filename in os.listdir(deepdrug3d_voxel_data):
    protein_name = filename[0:-4]
    full_path = voxel_folder + '/' + filename
    temp = np.load(full_path)
    voxel[cnt,:] = temp
    if protein_name in atps:
        label[cnt] = 0
    elif protein_name in hemes:
        label[cnt] = 1
    else:
        print protein_name
        print 'Something is wrong...'
        break
    cnt += 1
'''
