#文件导入
import numpy as np 
import cv2 
import os 
import matplotlib.pyplot as plt 
import time 
import os.path as osp 
import xml.etree.ElementTree as ET 
from matplotlib.colors import LogNorm



## == 批量处理 ==##
# 读取图片
def Get_rectangle_loc(num_IMG = 1):
    fore_name =['OriRGB_pitaya_', 'RbRGB_pitaya_']
    image_name = fore_name[0]+ str(num_IMG).zfill(6)

    rootpath1 ='VOCdevkit/VOC2007/JPEGImages/'
    image = cv2.imread( os.path.join(rootpath1, image_name +'.jpg'))        
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     
    # plt.rcParams[ 'figure.figsize' ] = [16, 6]                           
    # plt.imshow(image) 
    
    # 去除背景的图像
    rootpath_Rb = 'JPEGImages_Rb_HLGMulti/'
    image_name_Rb = fore_name[1]+ str(num_IMG).zfill(6)
    # print('image_name_Rb',image_name_Rb)
    # print('path',os.path.join(rootpath1, image_name_Rb +'.jpg'))
    image_Rb = cv2.imread( os.path.join(rootpath_Rb, image_name_Rb +'.jpg')) 
    # image2 = cv2.cvtColor(image_Rb, cv2.COLOR_BGR2RGB) 
    # plt.rcParams[ 'figure.figsize' ] = [16, 6]                           
    # plt.imshow(image2)    

    # 读取 xml 
    # 读取路径下的文件  
    rootpath2 = 'VOCdevkit/VOC2007/Annotations/'
    xml_filepath = os.path.join(rootpath2, image_name +'.xml')       
    # print(xml_filepath) 

    updateTree = ET.parse(xml_filepath)  # parse xml documents into element tree
    root = updateTree.getroot()      # return root element


    # dict - bbox - cls 
    rectangle_loc={ }
    class_name = ["NO", "OWF", "OF", "FCC", "OB"]
    for cls in class_name:
        rectangle_loc[cls] = []


    for obj in root.iter('object'):
        cls = obj.find('name').text

        if cls == 'NO':
            xmlbox = obj.find('bndbox') 
            b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
            rectangle_loc['NO'].append(b)    

        elif cls == 'OWF':
            xmlbox = obj.find('bndbox') 
            b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
            rectangle_loc['OWF'].append(b) 

        elif cls == 'OF':
            xmlbox = obj.find('bndbox') 
            b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
            rectangle_loc['OF'].append(b)

        elif cls == 'FCC':
            xmlbox = obj.find('bndbox') 
            b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
            rectangle_loc['FCC'].append(b) 

        elif cls == 'OB':
            xmlbox = obj.find('bndbox') 
            b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
            rectangle_loc['OB'].append(b) 
        
        else:
            print( 'no target' ) 

    # print(rectangle_loc, image_name, image_name_Rb)
    # return rectangle_loc, image, image_name, image_Rb, image_name_Rb
    return image_name, rectangle_loc

# 找出同时包含5个类别的图片
num_IMG_find=[]
for num_IMG in range(1,1300):
    image_name, rectangle_loc = Get_rectangle_loc(num_IMG = num_IMG)
    
    No_none_val_list = [] 
    for key_cls, value_rectangle in rectangle_loc.items():
        # print(key_cls, value_rectangle) 
        if value_rectangle != []:
            No_none_val_list.append(1)
    if sum(No_none_val_list)==5: # 5个类别都包含==5
        print('satisfied image',image_name)
        num_IMG_find.append(num_IMG)
print('num_IMG_find', num_IMG_find)
