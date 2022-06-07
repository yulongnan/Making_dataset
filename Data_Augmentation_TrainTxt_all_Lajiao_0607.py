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
from tqdm import tqdm

### ==== train_origin_txt 
txt_train_origin_path = 'ImageSets\Main\origintrain.txt' 
fr_train = open(txt_train_origin_path,'r') 
train_name = [] 
for l in fr_train.readlines():  
    lines3 = l.split() 
    train_name.append(lines3[0]) 
# print(train_name)

train_name_all = []
### === 添加数据增强-图片名
train_name_AUG=[]
for i_train_name in tqdm(train_name):
    end_name = i_train_name[-4:] 

    path_match = 'IMG_AUG\Annotations_AUG\\' + '*'+ end_name + ".xml"  # 定义--数据增强后的文件名
    pres = glob.glob(path_match)
    for i_pres in (pres):
        i_pres_basename = os.path.basename(i_pres).split('.')[0] 
        # print(i_pres_basename)
        train_name_all.append(i_pres_basename)
        train_name_AUG.append(i_pres_basename)
print('训练图片数量=', len(train_name_all))
print('Ori-训练图片数量=', len(train_name)) 
print('AUG-训练图片数量(含原始)=', len(train_name_AUG)) 

### ==== 写入txt
saveBasePath=r"ImageSets/Main/" # 定义--数据增强后的traintxt--路径
pre_str = 'AUG_OFAll' 
ftrain = open(os.path.join(saveBasePath,pre_str +'train.txt'), 'w') 
for i in train_name_all: 
    name = i +'\n' 
    ftrain.write(name) 
ftrain.close()

### ==== 计算类别
def parse_obj(xml_path, filename):
    tree=ET.parse(xml_path + filename)
    objects=[]
    for obj in tree.findall('object'):
        obj_struct={}
        obj_struct['name']=obj.find('name').text
        objects.append(obj_struct)
    return objects


xml_path = 'IMG_AUG\Annotations_AUG\\'     # xml位置 
filenames = train_name_all                           # 文件名 

def Cal_class_number(xml_path, filenames): # xml位置 'Annotations\\' filenames=['a1','a2', ]
    recs={}
    obs_shape={}
    classnames=[]
    num_objs={}
    obj_avg={}

    list_cls_pitaya_filename = []  # 查找错误标记类
    list_cls_OF_filename = []      # 查找OF
    list_cls_FCC_filename = []     # 查找FCC

    for i,name in enumerate(filenames): #读所有文件
        recs[name]=parse_obj(xml_path, name + '.xml' )
    # print(recs)
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

# 计算类别数
print(' All数据集(含原始) -ALL ')
Cal_class_number(xml_path, filenames = train_name_all )

print('###============= 分割线 ===============###')

print(' 数据集 -Original ')
Cal_class_number(xml_path, filenames=train_name)


### ==== test_origin_txt 
txt_test_origin_path = 'ImageSets\Main\originval.txt' 
fr_test = open(txt_test_origin_path,'r') 
test_name = [] 
for l in fr_test.readlines():  
    lines4 = l.split() 
    test_name.append(lines4[0]) 
# print(test_name, '\n', len(test_name))



test_name_all = []
for i_test_name in tqdm(test_name):
    end_name = i_test_name[-4:] 

    path_match = 'IMG_AUG\Annotations_AUG\\' + '*'+ end_name + ".xml"  # 定义--数据增强后的文件名
    pres = glob.glob(path_match) 
    for i_pres in (pres): 
        i_pres_basename = os.path.basename(i_pres).split('.')[0] 
        test_name_all.append(i_pres_basename)
print('测试图片数量=', len(test_name_all))

print('###============= 分割线 ===============###')
print(' 数据集 -Test ')
Cal_class_number(xml_path, filenames=test_name_all)


