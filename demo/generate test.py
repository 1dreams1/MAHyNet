import glob
import numpy as np

import glob
import numpy as np

#allFileFaList = glob.glob(r"F:\Download\lengent\venv\ePooling-master\RNAProt_supplementary_data\set2_add_feat_rnaprot_train_in30\*")
allFileFaList = glob.glob(r"F:\Download\lengent\venv\MAHyNet-main\RNAProt_supplementary_data\set1_add_feat_rnaprot_train_in23\*")
#print(allFileFaList)
def create_test(allFileFaList):
    sh = 0

    for FilePath in allFileFaList:
        count = 0
        filenames = glob.glob(FilePath+"\\trainr.test")
        for allFileFa in filenames:
            AllTem = allFileFa.split("\\")[-1].split(".")[-2]#negatives
            output_dir = allFileFa.split(AllTem)[0]#route
            print(output_dir)
            f1 = open(allFileFa, 'r')
            f2 = open(output_dir + 'test.data', 'w')
            f3= open(output_dir + 'train.data', 'w')
            lines=f1.readlines()
            Allseq = np.random.randint(0, len(lines), len(lines))
            p=int(len(lines)*0.8)
            for i in Allseq:
                if (count<=p):
                    f3.writelines(lines[i])
                    count += 1
                else:
                    f2.writelines(lines[i])



create_test(allFileFaList)
