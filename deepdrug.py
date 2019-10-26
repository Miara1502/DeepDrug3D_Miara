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

