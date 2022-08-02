import numpy as np
import pandas as pd
import h5py
import glob
import os
sha=0
counta=0
sh=0
count = 0

#You need modify path
allFileFaList = glob.glob(r"F:\Download\lengent\venv\MAHyNet-main\RNAProt_supplementary_data\set1_add_feat_rnaprot_train_in23\*")
#filenames = glob.glob(r"F:\Download\lengent\venv\ePooling-master\data\cnn\motif_discovery\wgEncodeAwg*\*.data")
#allFileFaList = glob.glob(r"F:\Download\lengent\venv\ePooling-master\demo\motif_discovery\wgEncodeAwg*")
print(allFileFaList)
def process_data(allFileFaList ):
    for FilePath in allFileFaList:
        filenames = glob.glob(FilePath+"\*.fa")
        print(filenames,'$$$$')
        for allFileFa in filenames:
            AllTem = allFileFa.split("\\")[-1].split(".")[-2]#negatives
            output_dir = allFileFa.split(AllTem)[0]#route
            SeqLen = 81
            f1 = open(allFileFa, 'r')
            #print(s)
            c=0
            keyword='>'
            if (AllTem == 'negatives'):
                f2 = open(output_dir + 'negatives1.fa', 'w')
                #f3 = open(output_dir + 'train2.data', 'w')
                for line in f1:
                    if keyword in line:
                        line = line.replace(">", "\n>")
                        f2.writelines(line)
                    else:
                        line=line.replace("\n","")
                        f2.writelines(line)


process_data(allFileFaList)

