#---------------------------------------------#
#   运行前一定要修改classes
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#---------------------------------------------#
import xml.etree.ElementTree as ET
from os import getcwd
import numpy as np 
import os
from utils.utils import get_classes


VOCdevkit_path = '..\datasets\VOCdevkit'
sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
#-----------------------------------------------------#
#   这里设定的classes顺序要和model_data里的txt一样
#-----------------------------------------------------#
# #classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# classes = ["NO", "OWF", "OF", "FCC", "OB"]

classes_path        = 'model_data/voc_classes.txt'
classes, _      = get_classes(classes_path)

def voc2frcnn(train_percent = 0.7,val_percent = 0.15, test_percent = 0.15):  #设置函数训练、验证、测试数据集比例
    
    np.random.seed(1)  # 定义随机数种子
    
    # 路径   
    xmlfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
    saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
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

    datasetindex = np.arange(1, num_Total+1, 1) 
    print('datasetindex',datasetindex, len(datasetindex ))  

    train_index = np.random.choice(datasetindex, num_train, replace = False)   # replace = False 保证元素不重复
    print('train_index',train_index, len(train_index))   

    datasetindex_remian= list(set(datasetindex) ^ set(train_index)) #剩余的数据集索引   
    print('datasetindex_remian',datasetindex_remian, len(datasetindex_remian))   

    val_index = np.random.choice(datasetindex_remian, num_val, replace = False)   # replace = False 保证元素不重复
    print('val_index', val_index, len(val_index))  

    test_index = list( set(datasetindex_remian) ^ set(val_index) ) 
    print('test_index', test_index,len(test_index) ) 

    ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    

    #索引相应xml文件名到txt

    for i in train_index:
        name=total_xml[i-1][:-4]+'\n'
        ftrain.write(name)    
    for i in val_index: 
        name=total_xml[i-1][:-4]+'\n'
        fval.write(name) 
    for i in test_index:
        name=total_xml[i-1][:-4]+'\n' 
        ftest.write(name) 
        

    ftrain.close()  
    fval.close()  
    ftest .close()

def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1: 
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


wd = getcwd()
def voc_annotation(): # 生成VOC数据集txt索引
    for year, image_set in sets:
        image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()        
        list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8') 
        for image_id in image_ids: 
            list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(os.path.abspath(VOCdevkit_path), year, image_id))
            convert_annotation(year, image_id, list_file)
            list_file.write('\n')
        list_file.close()

def voc_dataset_create(train_percent = 0.7,val_percent = 0.15, test_percent = 0.15):
    voc2frcnn(train_percent = train_percent  ,val_percent =val_percent , test_percent=test_percent) # 设置函数训练、验证、测试数据集比例 
    voc_annotation() # 生成VOC数据集  2007_XXX.txt索引

if __name__ == "__main__": 
    #创建数据集
    voc_dataset_create(train_percent = 0.7,val_percent = 0.15, test_percent = 0.15)
