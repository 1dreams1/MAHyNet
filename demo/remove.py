import numpy as np
import pandas as pd
import h5py
import glob
import sys
import os
import time
#allFileFaList = glob.glob(r"F:\Download\lengent\venv\ePooling-master\RNAProt_supplementary_data\set2_add_feat_rnaprot_train_in30\*")
allFileFaList = glob.glob(r"F:\Download\lengent\venv\MAHyNet-main\RNAProt_supplementary_data\set1_add_feat_rnaprot_train_in23\*")
#print(allFileFaList)
def remove(allFileFaList):
    for FilePath in allFileFaList:
        filenames = glob.glob(FilePath + "\\*")
        for allFileFa in filenames:
            AllTem = allFileFa.split("\\")[-1].split(".")[-1]  # negatives
            print(AllTem )
            if  AllTem!='fa' :
                os.remove(allFileFa)
    for FilePath in allFileFaList:
        filenames1 = glob.glob(FilePath + "\\train.fa")
        for allFileFa1 in filenames1:
            os.remove(allFileFa1)



remove(allFileFaList)
