import glob
import numpy as np



#You need modify path
allFileFaList = glob.glob(r"F:\Download\lengent\venv\MAHyNet-main\RNAProt_supplementary_data\set2_add_feat_rnaprot_train_in30\*")

def create_laber(allFileFaList):
    sh = 0
    count = 0
    for FilePath in allFileFaList:

        filenames = glob.glob(FilePath+"\*.fa")
        #print(filenames,'$$$$')
        for allFileFa in filenames:
            AllTem = allFileFa.split("\\")[-1].split(".")[-2]#negatives
            output_dir = allFileFa.split(AllTem)[0]#route
            print(output_dir)

            SeqLen = 81
            #print(allFileFa,'@@@')
            #print(AllTem,output_dir)
            c=0
            keyword='>'
            if (AllTem == 'negatives'):
                f1 = open(allFileFa, 'r')
                labal = ' 0'+'\n'
                f2 = open(output_dir + 'train0.txt', 'w')
                #f3 = open(output_dir + 'train2.data', 'w')
                for line in f1:
                    #print(line)
                    if keyword in line:
                        line = line.replace("\n", " ")
                        f2.writelines(line)
                    else:
                        line = line.replace("\n", labal)
                        f2.writelines(line)
                f1.close()
                f2.close()

            else:
                fa = open(allFileFa, 'r')
                labal = ' 1' + '\n'
                fb = open(output_dir + 'train1.txt', 'w')
                # f3 = open(output_dir + 'train2.data', 'w')
                for linea in fa:
                    #print(line)
                    if keyword in linea:
                        linea = linea.replace("\n", " ")
                        fb.writelines(linea)
                    else:
                        linea = linea.replace("\n", labal)
                        fb.writelines(linea)
                fa.close()
                fb.close()


#merge
        filenames1 = glob.glob(FilePath+"\*.txt")
        #print( FilePath,'$$$$')
        for allFileFa1 in filenames1:
            AllTem = allFileFa1.split("\\")[-1].split(".")[0]
            output_dir = allFileFa1.split(AllTem)[0]
            #print(AllTem, '**')
            #print(output_dir,'%%')

            if ( AllTem=='train0' or 'train1'):
                #print(AllTem)
                ft1 = open(allFileFa1, 'r')
                ft2 = open(output_dir + 'trainr.test', 'a')
                s1 = ft1.readlines()
                th = len(s1)
                if(th<5000000):
                    ft2.seek(2)
                    ft2.writelines(s1)
                    sh += th
                    count+=1
                print(count)
                print(sh)




def rewrite(allFileFaList):
    for FilePath in allFileFaList:

        filenames = glob.glob(FilePath+"\\trainr.test")
        #print(filenames,'$$$$')
        for allFileFa in filenames:
            AllTem = allFileFa.split("\\")[-1].split(".")[-2]#negatives
            output_dir = allFileFa.split(AllTem)[0]#route
            fg = open(allFileFa, 'r')
            f = open(output_dir + 'train.fa', 'w')
            lines=fg.readlines()
            Allseq = np.random.randint(0, len(lines), len(lines))
            for i in  Allseq:
                f.writelines(lines[i])



create_laber(allFileFaList)
rewrite(allFileFaList)