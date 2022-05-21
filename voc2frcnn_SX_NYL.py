#----------------------------------------------------------------------#
#   验证集的划分在train.py代码里面进行
#   test.txt和val.txt里面没有内容是正常的。训练不会使用到。
#----------------------------------------------------------------------#
'''
#--------------------------------注意----------------------------------#
如果在pycharm中运行时提示：
FileNotFoundError: [WinError 3] 系统找不到指定的路径。: './VOCdevkit/VOC2007/Annotations'
这是pycharm运行目录的问题，最简单的方法是将该文件复制到根目录后运行。
可以查询一下相对目录和根目录的概念。在VSCODE中没有这个问题。
#--------------------------------注意----------------------------------#
'''
import os
import random
import numpy as np 

np.random.seed(1)  # 定义随机数种子

xmlfilepath=r'./Annotations_X1_1100/'    
saveBasePath=r"./ImageSets/Main/"   
 
#----------------------------------------------------------------------#
#   设置训练、验证、测试数据集比例   
#   修改  train_percent = 0.7    val_percent = 0.15    test_percent = 0.15 
#----------------------------------------------------------------------#

#设置训练、验证、测试数据集比例
# train_percent = 0.7 
# val_percent = 0.15 
# test_percent = 0.15 

if not os.path.isdir(saveBasePath):
    os.makedirs(saveBasePath)


def voc2frcnn(train_percent = 0.7,val_percent = 0.15, test_percent = 0.15):  #设置函数训练、验证、测试数据集比例
    #读取xml文件
    temp_xml = os.listdir(xmlfilepath)
    total_xml = []
    for xml in temp_xml:
        if xml.endswith(".xml"):  
            total_xml.append(xml)   

    #计算xml文件总数   
    num_Total=len(total_xml)  

    #计算训练、验证、测试数据集
    num_train = int(round(train_percent * num_Total)) #2021.09.02
    num_val = int(val_percent * num_Total)
    num_test = int(test_percent * num_Total)

    train_index  = temp_xml[:num_train]
    val_index = temp_xml[num_train: num_train + num_val]
    test_index = temp_xml[num_train + num_val:]

    # datasetindex = np.arange(1, num_Total+1, 1) 
    # print('datasetindex',datasetindex, len(datasetindex ))  

    # train_index = np.random.choice(datasetindex, num_train, replace = False)   # replace = False 保证元素不重复
    # print('train_index',train_index, len(train_index))   

    # datasetindex_remian= list(set(datasetindex) ^ set(train_index)) #剩余的数据集索引   
    # print('datasetindex_remian',datasetindex_remian, len(datasetindex_remian))   

    # val_index = np.random.choice(datasetindex_remian, num_val, replace = False)   # replace = False 保证元素不重复
    # print('val_index', val_index, len(val_index))  

    # test_index = list( set(datasetindex_remian) ^ set(val_index) ) 
    # print('test_index', test_index,len(test_index) ) 

    ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    

    #索引相应xml文件名到txt

    # for i in train_index:
    #     name=total_xml[i-1][:-4]+'\n'
    #     ftrain.write(name)    
    # for i in val_index: 
    #     name=total_xml[i-1][:-4]+'\n'
    #     fval.write(name) 
    # for i in test_index:
    #     name=total_xml[i-1][:-4]+'\n' 
    #     ftest.write(name) 

    for i in train_index:
        name=i+'\n'
        ftrain.write(name)    
    for i in val_index: 
        name=i+'\n'
        fval.write(name) 
    for i in test_index:
        name=i+'\n' 
        ftest.write(name)    

    print('train_data_num = ', len(train_index))
    print('val_data_num = ', len(val_index))
    print('test_data_num = ', len(test_index))

    ftrain.close()  
    fval.close()  
    ftest .close()

if __name__ == "__main__": 
    voc2frcnn(train_percent = 0.8,val_percent = 0.2, test_percent = 0.0) #设置函数训练、验证、测试数据集比例
    # 注意：line np.random.seed(1)  # 定义随机数种子
