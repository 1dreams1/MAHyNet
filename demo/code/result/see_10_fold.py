import glob
import sys
import os
import time
import numpy as np
#You need modify path
allFileFaList = glob.glob(r"F:\Download\lengent\venv\MAHyNet-main\demo\result_sum\*")
result_path = r"F:\Download\lengent\venv\MAHyNet-main\demo\code\result\result_sum.txt"

def see(allFileFaList):
    cour = []
    cour30 = []
    count = 0
    countauc=0
    countnumber=0
    f = open(result_path, 'w')
    #file=[]
    for FilePath in allFileFaList:
        filenames = glob.glob(FilePath + "\\*.npy")
        count+=1
        countnumber += 1
        #print(filenames,'\n',count)

        data = FilePath.split("\\")[-1]
        for allFileFa in filenames:
            countauc+=1
            AUC = np.load( allFileFa, encoding='bytes', allow_pickle=True)
            auc=str(AUC) +'  '+ str(count) +' '+ str(data)+'\n'
            print(auc)
            #file.extend(auc)
            cour.append(AUC)
            AUCs = np.mean(cour)
            f.writelines(auc)

            if data == 'PZC3H7B_Baltz2012_gt_out':
                au=float(AUCs)
                auc23 = '平均auc23:' + str(AUCs) + '\n'
                print(auc23)
                f.writelines(auc23)

            elif data == 'ZUPF1_K562_gt_out':
                ac=float(AUCs)
                auc53 = '平均auc53:' + str(AUCs) + '\n'
                print(auc53)
                f.writelines(auc53)
            if countnumber > 23:
                cour30.append(AUC)
                AUC30= np.mean(cour30)
                if  countnumber==53:
                    auc30 = '平均auc30:' + str(AUC30) + '\n'
                    print(auc30)
                    f.writelines(auc30)


    f.close()


see(allFileFaList)