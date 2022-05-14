# 获得训练数据集数据增强后的train.txt

import os
import random
import numpy as np 
import glob
import os
import xml.etree.ElementTree as ET
import numpy as np
# np.set_printoptions(suppress=True, threshold=np.nan)
import matplotlib
from PIL import Image
import shutil
import tqdm

### ==== train_origin_txt 
txt_train_origin_path = 'VOCdevkit\VOC2007\ImageSets\Main\origintrain.txt'
fr_train = open(txt_train_origin_path,'r')
train_name = []
for l in fr_train.readlines(): 
    lines3 = l.split() 
    train_name.append(lines3[0]) 
# print(train_name)

train_name_all = []
### === 添加数据增强AUG-图片名
train_name_AUG = []
for i_train_name in train_name:
    end_name = i_train_name[-6:]

    path_match = 'VOCdevkit\VOC2007\Annotations_AUG\\' + '*'+ end_name + ".xml"  # 定义--数据增强后的文件名
    pres = glob.glob(path_match) 
    for i_pres in (pres):
        i_pres_basename = os.path.basename(i_pres).split('.')[0] 
        # print(i_pres_basename) 
        train_name_all.append(i_pres_basename) 
        train_name_AUG.append(i_pres_basename)
print('AUG训练图片数量=', len(train_name_all)) 

### === 添加OF独立增强数据集--图片名
train_name_OF =[] 
for i_train_name in train_name:
    end_name = i_train_name[-6:]

    path_match = 'VOCdevkit\VOC2007\Annotations_AUG_OF\\' + '*'+ end_name + ".xml"  # 定义--数据增强后的文件名
    pres = glob.glob(path_match) 
    for i_pres in (pres):
        i_pres_basename = os.path.basename(i_pres).split('.')[0] 
        # print(i_pres_basename) 
        train_name_all.append(i_pres_basename) 
        train_name_OF.append(i_pres_basename)
print('AUG-OF-训练图片数量=', len(train_name_all)) 
print('AUG-训练图片数量=', len(train_name_AUG)) 
print('Ori-训练图片数量=', len(train_name)) 
print('OF-训练图片数量=', len(train_name_OF)) 
### ==== 写入txt
saveBasePath=r"./VOCdevkit/VOC2007/ImageSets/Main/" # 定义--数据增强后的traintxt--路径
pre_str = 'AUG_OF_' 
ftrain = open(os.path.join(saveBasePath,pre_str +'train.txt'), 'w') 

for i in train_name_all: 
    name = i +'\n'
    ftrain.write(name) 
ftrain.close()

pre_str = 'OF_'
ftrain_OF = open(os.path.join(saveBasePath,pre_str +'train.txt'), 'w') 
for i in train_name_OF: 
    name = i +'\n'
    ftrain_OF.write(name) 
ftrain_OF.close()

### ==== 计算类别
def parse_obj(xml_path, filename):
    tree=ET.parse(xml_path + filename)
    objects=[]
    for obj in tree.findall('object'):
        obj_struct={}
        obj_struct['name']=obj.find('name').text
        objects.append(obj_struct)
    return objects


xml_path = 'VOCdevkit\VOC2007\Annotations_AUG\\' # xml位置 
xml_path_2 = 'VOCdevkit\VOC2007\Annotations_AUG_OF\\'
filenames = train_name_all                           # 文件名 

def Cal_class_number(xml_path, xml_path_2, filenames): # xml位置 'Annotations\\' filenames=['a1',a2, ]
    recs={}
    obs_shape={}
    classnames=[]
    num_objs={}
    obj_avg={}

    list_cls_pitaya_filename = []  # 查找错误标记类
    list_cls_OF_filename = []  # 查找OF
    list_cls_FCC_filename = [] # 查找FCC

    for i,name in enumerate(filenames): #读所有文件
        if os.path.isfile(os.path.join(xml_path, name + '.xml')):
            recs[name]=parse_obj(xml_path, name + '.xml' )
        else:
            recs[name]=parse_obj(xml_path_2, name + '.xml' )

        
    
    for name in filenames:  #filenames[:1]
        for object in recs[name]:
            # print(object) object = {'name': 'OF'}
            if object['name'] not in num_objs.keys():
                num_objs[object['name']]=1
            else:
                num_objs[object['name']]+=1

            if object['name'] == 'pitaya': # 查找错误标记类
                list_cls_pitaya_filename.append(name) 
            if object['name'] == 'OF':     # 查找OF标记类
                list_cls_OF_filename.append(name) 
            if object['name'] == 'FCC':    # 查找FCC标记类
                list_cls_FCC_filename.append(name)

            if object['name'] not in classnames:
                classnames.append(object['name'])
    for name in classnames:
        print('{}:{}个'.format(name,num_objs[name]))

    print('信息统计算完毕。') 
    print(list_cls_pitaya_filename) 
    print('数据集计数', num_objs)

print('AUG + OF 数据集')
Cal_class_number(xml_path,xml_path_2 ,filenames=train_name_all)

train_name_AUG
print(' 数据集 -AUG ')
Cal_class_number(xml_path,xml_path_2 ,filenames=train_name_AUG)

print(' 数据集 -OF ')
Cal_class_number(xml_path,xml_path_2 ,filenames=train_name_OF)

print(' 数据集 -Original ')
Cal_class_number(xml_path,xml_path_2 ,filenames=train_name)