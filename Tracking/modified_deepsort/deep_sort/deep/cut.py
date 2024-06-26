import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom
import argparse


def main():
    img_path = 'D:\\deepsort\\Yolov5_StrongSORT_OSNet-3.0\\yolov5\\my_data\\images\\test\\'
    anno_path = 'D:\\train\\xml test\\'
    cut_path = 'D:\\deepsort\\Yolov5_StrongSORT_OSNet-3.0\\deep_sort_pytorch\\deep_sort\\deep\\cells\\test\\'
    if not os.path.exists(cut_path):
        os.makedirs(cut_path)
    imagelist = os.listdir(img_path)
    for image in imagelist:
        image_pre, ext = os.path.splitext(image)
        img_file = img_path + image
        img = cv2.imread(img_file)
        xml_file = anno_path + image_pre + '.xml'

        tree = ET.parse(xml_file)
        root = tree.getroot()
        obj_i = 0
        for obj in root.iter('object'):
            obj_i += 1
            cls = obj.find('name').text
            xmlbox = obj.find('bndbox')
            b = [int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
                 int(float(xmlbox.find('xmax').text)),
                 int(float(xmlbox.find('ymax').text))]
            img_cut = img[b[1]:b[3], b[0]:b[2], :]
            path = os.path.join(cut_path, cls)
            mkdirlambda = lambda x: os.makedirs(x) if not os.path.exists(x) else True
            mkdirlambda(path)
            if img_cut.size == 0:
                pass
            else:
                cv2.imwrite(os.path.join(cut_path, cls, '{}_{:0>2d}.jpg'.format(image_pre, obj_i)), img_cut)
                print("&&&&")


if __name__ == '__main__':
    main()
