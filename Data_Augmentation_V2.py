import os
import cv2
import xml.dom.minidom
from xml.dom.minidom import Document
import math


# 获取路径下所有文件的完整路径，用于读取文件用
def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles


def limit_value(a, b):
    if a < 1:
        a = 1
    if a > b:
        a = b - 1
    return a


# 读取xml文件，xmlfile参数表示xml的路径
def readXml(xmlfile):
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement
    sizelist = annotation.getElementsByTagName('size')  # [<DOM Element: filename at 0x381f788>]
    heights = sizelist[0].getElementsByTagName('height')
    height = int(heights[0].childNodes[0].data)
    widths = sizelist[0].getElementsByTagName('width')
    width = int(widths[0].childNodes[0].data)
    depths = sizelist[0].getElementsByTagName('depth')
    depth = int(depths[0].childNodes[0].data)
    objectlist = annotation.getElementsByTagName('object')
    bboxes = []
    for objects in objectlist:
        namelist = objects.getElementsByTagName('name')
        class_label = namelist[0].childNodes[0].data
        bndbox = objects.getElementsByTagName('bndbox')[0]
        x1_list = bndbox.getElementsByTagName('xmin')
        x1 = int(float(x1_list[0].childNodes[0].data))
        y1_list = bndbox.getElementsByTagName('ymin')
        y1 = int(float(y1_list[0].childNodes[0].data))
        x2_list = bndbox.getElementsByTagName('xmax')
        x2 = int(float(x2_list[0].childNodes[0].data))
        y2_list = bndbox.getElementsByTagName('ymax')
        y2 = int(float(y2_list[0].childNodes[0].data))
        # 这里我box的格式【xmin，ymin，xmax，ymax，classname】
        bbox = [x1, y1, x2, y2, class_label]
        bboxes.append(bbox)
    return bboxes, width, height, depth


# 图像旋转用，里面的angle是角度制的
def im_rotate(im, angle, center=None, scale=1.0):
    h, w = im.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    im_rot = cv2.warpAffine(im, M, (w, h))
    return im_rot


def im_flip(im, method='H'):  # 翻转图像
    if method == 'H':  # Flipped Horizontally 水平翻转
        im_flip = cv2.flip(im, 1)
    elif method == 'V':  # Flipped Vertically 垂直翻转
        im_flip = cv2.flip(im, 0)
    # elif method == 'HV':# Flipped Horizontally & Vertically 水平垂直翻转
    #    im_flip = cv2.flip(im, -1)
    return im_flip


def flip_image(imgs_path, method, anno_new_dir_, img_new_dir):
    j=1
    for img_path in imgs_path:
        # print(img_path)
        # 读取原图像

        j+=1
        # print(anno_new_dir)
        im = cv2.imread(img_path)
        flip_img = im_flip(im, method)  # 翻转
        (H, W, D) = flip_img.shape  # 得到翻转后的图像的高、宽、深度，用于书写xml
        file_name = os.path.basename(os.path.splitext(img_path)[0])    # 得到原图的名称
        anno_new_dir = anno_new_dir_ + '\\'+ method + '_' +file_name   # XML 新名称 修改202111.28
        print(anno_new_dir)
        ext = os.path.splitext(img_path)[-1]  # 得到原图的后缀
        # 保存翻转后图像
        new_img_name = '%s_%s%s' % (method, file_name, ext)
        cv2.imwrite(os.path.join(img_new_dir, new_img_name), flip_img)  # 新的命名方式为H/V+原图名称
        # 读取anno标签数据，返回相应的信息 
        anno = os.path.join(anno_path, '%s.xml' % file_name) 
        [gts, w, h, d] = readXml(anno)
        gt_new = []
        for gt in gts:
            x1 = gt[0]  # xmin
            y1 = gt[1]  # ymin
            x2 = gt[2]  # xmax
            y2 = gt[3]  # ymax
            classname = str(gt[4])
            if method == 'H':
                x1 = w - 1 - x1  # xmax
                x2 = w - 1 - x2  # xmin
                x1 = limit_value(x1, w)
                x2 = limit_value(x2, w)
                gt_new.append([x2, y1, x1, y2, classname])
            elif method == 'V':
                y1 = h - 1 - y1  # ymax
                y2 = h - 1 - y2  # ymin
                y1 = limit_value(y1, h)
                y2 = limit_value(y2, h)
                gt_new.append([x1, y2, x2, y1, classname])
        # writeXml(anno_new_dir, new_img_name, W, H, D, gt_new)
        writeXml(anno_new_dir, new_img_name, W, H, D, gt_new)


def rotate_image(aglens, angle_rad, imgs_path, anno_new_dir_, img_new_dir):
    j = 0  # 计数用
    angle_num = len(angles)
    for img_path in imgs_path:
        # 读取原图像
        im = cv2.imread(img_path)
        for i in range(angle_num):
            gt_new = []
            im_rot = im_rotate(im, angles[i])  # 旋转
            (H, W, D) = im_rot.shape  # 得到旋转后的图像的高、宽、深度，用于书写xml
            file_name = os.path.basename(os.path.splitext(img_path)[0])  # 得到原图的名称
            ext = os.path.splitext(img_path)[-1]  # 得到原图的后缀
            # 保存旋转后图像
            new_img_name = 'R%s_%s%s' % (angles[i], file_name, ext)
            cv2.imwrite(os.path.join(img_new_dir, new_img_name), im_rot)  # 新的命名方式为P+角度+原图名称
            # 读取anno标签数据，返回相应的信息
            anno = os.path.join(anno_path, '%s.xml' % file_name)
            [gts, w, h, d] = readXml(anno)

            anno_new_dir = anno_new_dir_ + '\\' + 'R' +str(aglens[i]) + '_' + file_name   # XML 新名称 修改202111.28

            # 计算旋转后gt框四点的坐标变换
            [xc, yc] = [float(w) / 2, float(h) / 2] 
            for gt in gts:
                # 计算左上角点的变换
                x1 = (gt[0] - xc) * math.cos(angle_rad[i]) - (yc - gt[1]) * math.sin(angle_rad[i]) + xc
                if int(x1) <= 0: x1 = 1.0
                if int(x1) > w - 1: x1 = w - 1
                y1 = yc - (gt[0] - xc) * math.sin(angle_rad[i]) - (yc - gt[1]) * math.cos(angle_rad[i])
                if int(y1) <= 0: y1 = 1.0
                if int(y1) > h - 1: y1 = h - 1
                # 计算右上角点的变换
                x2 = (gt[2] - xc) * math.cos(angle_rad[i]) - (yc - gt[1]) * math.sin(angle_rad[i]) + xc
                if int(x2) <= 0: x2 = 1.0
                if int(x2) > w - 1: x2 = w - 1
                y2 = yc - (gt[2] - xc) * math.sin(angle_rad[i]) - (yc - gt[1]) * math.cos(angle_rad[i])
                if int(y2) <= 0: y2 = 1.0
                if int(y2) > h - 1: y2 = h - 1
                # 计算左下角点的变换
                x3 = (gt[0] - xc) * math.cos(angle_rad[i]) - (yc - gt[3]) * math.sin(angle_rad[i]) + xc
                if int(x3) <= 0: x3 = 1.0
                if int(x3) > w - 1: x3 = w - 1
                y3 = yc - (gt[0] - xc) * math.sin(angle_rad[i]) - (yc - gt[3]) * math.cos(angle_rad[i])
                if int(y3) <= 0: y3 = 1.0
                if int(y3) > h - 1: y3 = h - 1
                # 计算右下角点的变换
                x4 = (gt[2] - xc) * math.cos(angle_rad[i]) - (yc - gt[3]) * math.sin(angle_rad[i]) + xc
                if int(x4) <= 0: x4 = 1.0
                if int(x4) > w - 1: x4 = w - 1
                y4 = yc - (gt[2] - xc) * math.sin(angle_rad[i]) - (yc - gt[3]) * math.cos(angle_rad[i])
                if int(y4) <= 0: y4 = 1.0
                if int(y4) > h - 1: y4 = h - 1
                xmin = min(x1, x2, x3, x4)
                xmax = max(x1, x2, x3, x4)
                ymin = min(y1, y2, y3, y4)
                ymax = max(y1, y2, y3, y4)
                # 把因为旋转导致的特别小的 长线型的去掉
                # w_new = xmax-xmin+1
                # h_new = ymax-ymin+1
                # ratio1 = float(w_new)/h_new
                # ratio2 = float(h_new)/w_new
                # if(1.0/6.0<ratio1<6 and 1.0/6.0<ratio2<6 and w_new>9 and h_new>9):
                classname = str(gt[4])
                gt_new.append([xmin, ymin, xmax, ymax, classname])
                # 写出新的xml
                writeXml(anno_new_dir, new_img_name, W, H, D, gt_new)
            j = j + 1
            if j % 100 == 0: print('----%s----' % j)


# 写xml文件，参数中tmp表示路径，imgname是文件名（没有尾缀）ps有尾缀也无所谓
def writeXml(tmp, imgname, w, h, d, bboxes):
    doc = Document()
    # owner
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    # owner
    folder = doc.createElement('folder')
    annotation.appendChild(folder)
    folder_txt = doc.createTextNode("VOC2007")
    folder.appendChild(folder_txt)

    filename = doc.createElement('filename')
    annotation.appendChild(filename)
    filename_txt = doc.createTextNode(imgname)
    filename.appendChild(filename_txt)
    # ones#
    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    source.appendChild(database)
    database_txt = doc.createTextNode("My Database")
    database.appendChild(database_txt)

    annotation_new = doc.createElement('annotation')
    source.appendChild(annotation_new)
    annotation_new_txt = doc.createTextNode("VOC2007")
    annotation_new.appendChild(annotation_new_txt)

    image = doc.createElement('image')
    source.appendChild(image)
    image_txt = doc.createTextNode("flickr")
    image.appendChild(image_txt)
    # owner
    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid = doc.createElement('flickrid')
    owner.appendChild(flickrid)
    flickrid_txt = doc.createTextNode("NULL")
    flickrid.appendChild(flickrid_txt)

    ow_name = doc.createElement('name')
    owner.appendChild(ow_name)
    ow_name_txt = doc.createTextNode("idannel")
    ow_name.appendChild(ow_name_txt)
    # onee#
    # twos#
    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    size.appendChild(width)
    width_txt = doc.createTextNode(str(w))
    width.appendChild(width_txt)

    height = doc.createElement('height')
    size.appendChild(height)
    height_txt = doc.createTextNode(str(h))
    height.appendChild(height_txt)

    depth = doc.createElement('depth')
    size.appendChild(depth)
    depth_txt = doc.createTextNode(str(d))
    depth.appendChild(depth_txt)
    # twoe#
    segmented = doc.createElement('segmented')
    annotation.appendChild(segmented)
    segmented_txt = doc.createTextNode("0")
    segmented.appendChild(segmented_txt)

    for bbox in bboxes:
        # threes#
        object_new = doc.createElement("object")
        annotation.appendChild(object_new)

        name = doc.createElement('name')
        object_new.appendChild(name)
        name_txt = doc.createTextNode(str(bbox[4]))
        name.appendChild(name_txt)

        pose = doc.createElement('pose')
        object_new.appendChild(pose)
        pose_txt = doc.createTextNode("Unspecified")
        pose.appendChild(pose_txt)

        truncated = doc.createElement('truncated')
        object_new.appendChild(truncated)
        truncated_txt = doc.createTextNode("0")
        truncated.appendChild(truncated_txt)

        difficult = doc.createElement('difficult')
        object_new.appendChild(difficult)
        difficult_txt = doc.createTextNode("0")
        difficult.appendChild(difficult_txt)
        # threes-1#
        bndbox = doc.createElement('bndbox')
        object_new.appendChild(bndbox)

        xmin = doc.createElement('xmin')
        bndbox.appendChild(xmin)
        xmin_txt = doc.createTextNode(str(bbox[0]))
        xmin.appendChild(xmin_txt)

        ymin = doc.createElement('ymin')
        bndbox.appendChild(ymin)
        ymin_txt = doc.createTextNode(str(bbox[1]))
        ymin.appendChild(ymin_txt)

        xmax = doc.createElement('xmax')
        bndbox.appendChild(xmax)
        xmax_txt = doc.createTextNode(str(bbox[2]))
        xmax.appendChild(xmax_txt)

        ymax = doc.createElement('ymax')
        bndbox.appendChild(ymax)
        ymax_txt = doc.createTextNode(str(bbox[3]))
        ymax.appendChild(ymax_txt)

        print(bbox[0], bbox[1], bbox[2], bbox[3])

    xmlname = os.path.splitext(imgname)[0]
    tempfile = tmp + ".xml"
    print(tempfile)
    with open(tempfile, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
        f.close()
    return


if __name__ == '__main__':

    # 数据路径
    root = 'IMG_AUG'
    path_name2 = '_Rb_Sort'
    img_dir = 'JPEGImages' + path_name2
    anno_path = 'Annotations' + path_name2
    imgs_path = GetFileFromThisRootDir(img_dir)  # 返回每一张原图的路径
    AUG = 'Rotate'  # 数据扩增的方式 Rotate代表旋转, Flip表示翻转

    # 存储新的anno位置
    anno_new_dir = os.path.join(root, AUG, 'xml')
    if not os.path.isdir(anno_new_dir):
        os.makedirs(anno_new_dir)
    # 扩增后图片保存的位置
    img_new_dir = os.path.join(root, AUG, 'images')
    if not os.path.isdir(img_new_dir):
        os.makedirs(img_new_dir)

    if AUG == 'Rotate':
        # 旋转角的大小，正数表示逆时针旋转
        angles = [90]  # 角度im_rotate用到的是角度制 [5, 90, 180, 270, 355]
        angle_rad = [angle * math.pi / 180.0 for angle in angles]  # cos三角函数里要用到弧度制的
        # 开始旋转
        rotate_image(angles, angle_rad, imgs_path, anno_new_dir, img_new_dir)
    elif AUG == 'Flip':
        method = 'V'
        flip_image(imgs_path, method, anno_new_dir, img_new_dir)

