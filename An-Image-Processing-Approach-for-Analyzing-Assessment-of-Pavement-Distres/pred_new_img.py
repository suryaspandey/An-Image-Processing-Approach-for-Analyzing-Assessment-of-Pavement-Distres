# -*- coding: utf-8 -*-
"""
Created on Sat May 13 11:30:59 2017

@author: Surya
"""

from PIL import Image, ImageFile, ImageFilter, ImageEnhance
im = Image.open("road_2.jpg")

    #im = Image.open(path1 + '\\' + file)
    #print(file)
    # Blur
im =  im.resize((28,28))
im =  im.convert('L')
im1 = im.filter(ImageFilter.BLUR)
im2 = im.filter(ImageFilter.SHARPEN)
im3 = im.rotate(45)
im4 = im.transpose(Image.FLIP_LEFT_RIGHT)
im5 = im.transpose(Image.FLIP_TOP_BOTTOM)
im6 = ImageEnhance.Contrast(im).enhance(1.3)
im7=  im.filter(ImageFilter.MedianFilter(size=3))
im8=  im.filter(ImageFilter.ModeFilter(size=3))
im9=  im.filter(ImageFilter.EMBOSS)
im10= im.filter((ImageFilter.SMOOTH))
im11= im.filter(ImageFilter.CONTOUR)
    #im.show()
    
imgs = [im1,im2,im3,im4,im5,im6,im7,im8,im9,im10,im11]

immatrix_test = array([array(im2).flatten() for im2 in imgs])

test_test = immatrix_test.reshape(immatrix_test.shape[0] ,img_rows, img_cols, 1)

predict_test = model.predict_classes(test_test)
