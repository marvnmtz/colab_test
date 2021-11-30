# -*- coding: utf-8 -*-
import os
import cv2

def resize(img_src_dir, img_target_dir, img_list):
    for img_name in img_list:
        if img_name.split('.')[1] == 'png':
            try:
                img = cv2.imread(img_src_dir + "\\" + img_name)
                height, width, _ = img.shape
                if height != 299 or width != 299:
                    big_img = cv2.resize(img, (299, 299))
                    cv2.imwrite(img_target_dir + "\\" + img_name, big_img)
                else:
                    cv2.imwrite(img_target_dir + "\\" + img_name, img)
            except:
                print("error:")
                print(img_name)
    return 1