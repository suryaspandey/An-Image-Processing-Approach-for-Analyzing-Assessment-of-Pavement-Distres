# -*- coding: utf-8 -*-
"""
Created on Mon May 08 21:02:30 2017

@author: Surya
"""
import scipy.misc
from scipy import ndimage
import os
#import python exception
from PIL import Image, ImageFilter, ImageEnhance
#path1='C:\Users\Surya\Downloads\FlareGet\\non_crack_img'

path1= 'C:\Users\Surya\keras_cnn\keras_training\dummy_testing\c_dummy'

path2='C:\Users\Surya\keras_cnn\keras_training\dummy_testing\\test_aug'

listing = os.listdir(path1)

for file in listing:
   try:
    
   
    im = Image.open(path1 + '\\' + file)

    (name,ext) = os.path.splitext(file)
    #print(name)
    im.save(path2 + '\\' +name + "_org.JPEG")
   #print(file)
   
   # Blur
    im1 = im.filter(ImageFilter.BLUR)
    (name,ext) = os.path.splitext(file)
    im1.save(path2 + '\\' +name + "_blur.JPEG")
   
    print(name)
   # Sharpen
    im2 = im.filter(ImageFilter.SHARPEN)
    (name,ext) = os.path.splitext(file)
    im2.save(path2 + '\\' + name+"_sharp.JPEG")
   
   # Rotate
    im3 = im.rotate(45)
    (name,ext) = os.path.splitext(file)
    im3.save(path2 + '\\' + name+"_rot.JPEG")
   
   
   # flip left right
    im4 = im.transpose(Image.FLIP_LEFT_RIGHT)
    (name,ext) = os.path.splitext(file)
    im4.save(path2 + '\\' + name+"_LR.JPEG")
   
   
   # flip top bottom
    im5 = im.transpose(Image.FLIP_TOP_BOTTOM)
    (name,ext) = os.path.splitext(file)
    im5.save(path2 + '\\' + name+"_TB.JPEG")
   
   #Enhancement
    im6 = ImageEnhance.Contrast(im).enhance(1.3)
    (name,ext) = os.path.splitext(file)
    im6.save(path2 + '\\' + name+"_Enh.JPEG")
   
   
##    #median filter
    im7= im.filter(ImageFilter.MedianFilter(size=3))
    (name,ext) = os.path.splitext(file)
    im7.save(path2 + '\\' + name+"_MedFilt.JPEG")
##    
###    #Model filter
    im8= im.filter(ImageFilter.ModeFilter(size=3))
    (name,ext) = os.path.splitext(file)
    im8.save(path2 + '\\' + name+"_ModelFilt.JPEG")
###    
##    # Emboss
    im9= im.filter(ImageFilter.EMBOSS)
    (name,ext) = os.path.splitext(file)
    im9.save(path2 + '\\' + name+"_EMBOSS.JPEG")
##    
##
##    #Smooth
    im10=  im.filter((ImageFilter.SMOOTH))
    (name,ext) = os.path.splitext(file)
    im10.save(path2 + '\\' + name+"_Smooth.JPEG")
##    
##    #COnTOUR
    im11= im.filter(ImageFilter.CONTOUR)
    (name,ext) = os.path.splitext(file)
    im11.save(path2 + '\\' + name+"_CONTOUR.JPEG")
   
##    
    im12 = im.rotate(30)
    (name,ext) = os.path.splitext(file)
    im12.save(path2 + '\\' + name+"_rot30.JPEG")
#    
    im13 = im.rotate(270)
    (name,ext) = os.path.splitext(file)
    im13.save(path2 + '\\' + name+"_rot270.JPEG")
#    
    im14 = im.rotate(85)
    (name,ext) = os.path.splitext(file)
    im14.save(path2 + '\\' + name+"_rot85.JPEG")
#    
##     #Enhancement
    im15 = ImageEnhance.Contrast(im).enhance(1.5)
    (name,ext) = os.path.splitext(file)
    im15.save(path2 + '\\' + name+"_Enh3.JPEG")
    
      #Enhancement
    im16 = ImageEnhance.Contrast(im).enhance(2.3)
    (name,ext) = os.path.splitext(file)
    im16.save(path2 + '\\' + name+"_Enh2.JPEG")
    
  
    
    #padding
    longer_side = max(im.size)
    horizontal_padding = (longer_side - im.size[0]) / 2
    vertical_padding = (longer_side - im.size[1]) / 2
    im17 = im.crop(
            (
        -horizontal_padding,
        -vertical_padding,
        im.size[0] + horizontal_padding,
        im.size[1] + vertical_padding
                )
        )
    
    (name,ext) = os.path.splitext(file)
    im17.save(path2 + '\\' + name+"_Padding.JPEG")
    
     # Nearest neighbor (default):
    im18 = im.rotate(45, resample=Image.NEAREST)
    (name,ext) = os.path.splitext(file)
    im18.save(path2 + '\\' + name+"_NN.JPEG")
     
     # Linear interpolation:
    im19 = im.rotate(45, resample=Image.BILINEAR)
    (name,ext) = os.path.splitext(file)
    im19.save(path2 + '\\' + name+"_LI.JPEG")
     
     
     # Cubic spline interpolation:
    im20 = im.rotate(45, resample=Image.BICUBIC)
    (name,ext) = os.path.splitext(file)
    im20.save(path2 + '\\' + name+"_CI.JPEG")
   except:
     continue
# #%%import numpy as np
#import scipy.misc
#from scipy import ndimage
#import matplotlib.pyplot as plt
#
##im = scipy.misc.face(gray=True)
#lx, ly = im.shape
## Cropping
#im16 = im[lx/4:-lx/4, ly/4:-ly/4]
#im.show(im16)