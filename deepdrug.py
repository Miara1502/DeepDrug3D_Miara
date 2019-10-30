# -*- coding: utf-8  -*-

"""Module qui contient tous les fonctions utilisés pour la creéation des données
utilisés par le modèle Deepdrug3D
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
import glob
import random
#Lecture des datas :
def load_ligand(file):
    ''' read a ligand file and return a list as a np data frame
    '''
    list_ligand = np.loadtxt(file, dtype = str)
    return list_ligand

def load_voxel(voxel_files , file_number, nucleotide_list , hemes_list,
    steroid_list, control_list):
    ''' take a folder , a list of ligand and create
    4 .npY file which correspond of the data that the
    model is about to use ( features , target ,control
    and steroid)
    '''
    random.shuffle(voxel_files)
    targets= []
    features= []
    steroids = []
    for voxel in voxel_files[:file_number]:
        name = voxel[22:-4]
        if name in nucleotide_list:
            features.append(np.load(voxel))
            target = [1,0,0]
            #target = 0
            targets.append(target)
        elif name in hemes_list:
            features.append(np.load(voxel))
            target = [0,1,0]
            #target = 1
            targets.append(target)
        elif name in control_list:
            c = np.load(voxel)
            features.append(np.reshape(c,(14,32,32,32)))
            target = [0,0,1]
            #target = 2
            targets.append(target)
        elif name in steroid_list:
            steroid = np.load(voxel)
            steroids.append(np.reshape(steroid,(14,32,32,32)))
    features = np.array(features)
    targets = np.array(targets)
    steroids = np.array(steroids)
    np.save('features', features)
    np.save('targets', targets)
    np.save('steroides', steroids)
    print('features: ', features.shape)
    print('targets: ', targets.shape)
    print('stero: ', steroids.shape)


def main():
    ''' Main of the programm
    '''
    parser=argparse.ArgumentParser(description="Deepdrug traitement fichiers.")
    parser.add_argument("-n", "--nb-files",
                        help="Selectionner le nombre de fichiers voxel à utiliser",
                        type=int, default=200, dest="nFile")
    args = parser.parse_args()
    voxel_folder = 'deepdrug3d_voxel_data'
    files = glob.glob(voxel_folder+'/*.npy')
    nucleotides_list = load_ligand('nucleotide.list')
    hemes_list = load_ligand('heme.list')
    steroid_list = load_ligand('steroid.list')
    control_list = load_ligand('control.list')
    load_voxel(files, args.nFile, nucleotides_list , hemes_list , steroid_list,
    control_list)

if __name__ == "__main__":
    main()
