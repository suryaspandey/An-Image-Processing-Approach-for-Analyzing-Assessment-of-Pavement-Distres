# -*- coding: utf-8 -*-
"""


@author: Surya
"""

#%%
from keras.models import Sequential # import type of model
from keras.layers.core import Dense, Dropout, Activation, Flatten # import layers
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop,adam
from keras.preprocessing.image import ImageDataGenerator

#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import theano
from PIL import Image, ImageFile, ImageFilter, ImageEnhance
ImageFile.LOAD_TRUNCATED_IMAGES = True
from numpy import *

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split




#%%

# batch size to import
batch_size =128
# no of output classes
nb_classes = 3
# no of epoches  to train
nb_epoch = 20
# input image dimenssion
img_rows, img_cols = 28,28  
# no of convolutional filters to use
nb_filters = 24
# size of pooling area for max pooling
nb_pool = 2
# convolutional kernel size
nb_conv = 3
#%%


                
                #original dataset path
                path1='C:\Users\Surya\keras_cnn\keras_training\data_folder\crack_vs_all'

                #resized dataset path
                path2='C:\Users\Surya\keras_cnn\keras_training\data_folder\\all_data_resized_pot'
                
                listing = os.listdir(path1)
                #num_samples =size(listing)
                #print(num_samples)
                
                
                for file in listing:
                    (name,ext) = os.path.splitext(file)
                    #print(name)
                    im = Image.open(path1 + '\\' + file)
                    img = im.resize((img_rows,img_cols))
                    gray = img.convert('L')
                    gray.save(path2 + '\\' + file, "JPEG")

#    
imlist = os.listdir(path2)   
im1= array(Image.open('C:\Users\Surya\keras_cnn\keras_training\data_folder\\all_data_resized_pot' + '\\' + imlist[1])) # open one image to get size
m,n = im1.shape # get the size of the images
imnbr = len(imlist) # get the number of images
    
   #create matrix to store all flatnned images
immatrix = array([array(Image.open('C:\Users\Surya\keras_cnn\keras_training\data_folder\\all_data_resized_pot' + '\\'+im2)).flatten() for im2 in imlist], 'f')
 

#listing = os.listdir(path1)
num_samples =size(listing)
#print(num_samples)   
label= np.ones((num_samples,),dtype = int)
label[0:12182]=0 #crack
label[12182:20182]=1 #patch
label[20182:]=2 # pothole
#label[28267:]=3 # road
     
data,Label = shuffle(immatrix,label,random_state=4)
train_data = [data,Label]

img = immatrix[1600].reshape(img_rows,img_cols)
plt.imshow(img)
plt.imshow(img,cmap='gray')
print(train_data[0].shape)
print(train_data[1].shape)

(x,y)= (train_data[0],train_data[1])

#split x and y into training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=4)


y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
#    

x_train = x_train.reshape(x_train.shape[0] ,img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows , img_cols, 1)




x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255 # divide by maximum intensity pixels
x_test /= 255



print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#    
i = 6000
plt.imshow(x_train[i,0], interpolation= 'nearest')
print('label:', y_train[i,:])

#%% defining model architecture

model = Sequential()


#CONV=>RELU=>CONV=>CONV=>RELU=>MAXPOOL=>RELU=>FC
#model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
model.add(Conv2D(nb_filters,( nb_conv, nb_conv), border_mode='valid',
         input_shape =(img_rows, img_cols,1)))# input_shapes:no of channels
convout1 = Activation('tanh')
model.add(convout1)
model.add(Conv2D(nb_filters, (nb_conv, nb_conv))) #
model.add(Conv2D(24, (3, 3)))#
convout2 = Activation('tanh')#
model.add(convout2)#
model.add(MaxPooling2D(pool_size= (nb_pool,nb_pool)))
model.add(Dropout(0.25))
# Dropout acts as regulariser to avoid overfitting
#Extra 4 lines added
#model.add(Conv2D(24, (3, 3)))#
#convout3 = Activation('relu')#
#model.add(convout3)#
#model.add(MaxPooling2D(pool_size= (nb_pool,nb_pool)))


#set of FC layer: relu layers
model.add(Flatten())    
model.add(Dense(128))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))

#%%softmax classifier
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
from keras.optimizers import Adam
#Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)

#%%



#%% for training
model.fit(x_train, y_train,batch_size=batch_size,nb_epoch=nb_epoch,
           verbose=1, validation_data=(x_test,y_test))


#%%
hist = model.fit(x_train, y_train,batch_size=batch_size,nb_epoch=nb_epoch,
           verbose=1, validation_data=(x_test,y_test))
#hist.history will give accuracy from accuracy of epoch 1 to last epoch


#model.fit(x_train, y_train,nb_epoch=nb_epoch,
 #          verbose=1, validation_data=(x_test,y_test))

#%%#%% for training
hist = model.fit(x_train, y_train,batch_size=batch_size,nb_epoch=nb_epoch, 
          verbose=1,validation_split= 0.1)


#%% Graph Plotting

train_loss= hist.history['loss']
val_loss= hist.history['val_loss']
train_acc= hist.history['acc']
val_acc= hist.history['val_acc']
xc= range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('number of epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print plt.style.available
plt.style.use(['ggplot'])


plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('number of epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
print plt.style.available
plt.style.use(['ggplot'])

    #%% Confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
y_pred = model.predict(x_test)
print(y_pred)

#prdicts which image belongs to which class.. 
y_pred= np.argmax(y_pred,axis=1)
print(y_pred)

###########OR
y_pred= model.predict_classes(x_test)
print(y_pred)

p= model.predict_proba(x_test)
print(p)
target_names = ['CLASS0(CRACK)', 'CLASS1(PATCH)', 'CLASS2(ROAD)']


#
print(classification_report(np.argmax(y_test,axis=1),y_pred,target_names=target_names))
#diagonal values
print(confusion_matrix(np.argmax(y_test,axis=1),y_pred))

#%% saving weights
fname= "pav_dis_cnn.hdf5"
model.save_weights(fname,overwrite=True)

#%%loading weights
fname="pav_dis_cnn.hdf5"
model.load_weights(fname)


#%%
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score', score)
#print('Test accuracy',score)
predict= model.predict_classes(x_test[1:15])
print(y_test[1:15])

#%%
#from keras.utils.np_utils import to_categorical
#a= np.categorical_crossentropy(x_test, y_pred)
#%%



#%%
##another code
##visualising intermediate layers
#output_layer = model.layers[0].output
#output_fn= theano.function([model.layers[0].input], output_layer)
##
### input image
#input_img =x_train[0:1,:,:,:]
#print(input_img.shape)
##
##
##plt.show(input_img[0,0,:,:])
##plt.show(input_img[0,0,:,:])
##
#output_img = output_fn(input_img)
#print(output_img.shape)
#
#input_img = model.input
#layer_dict = dict([(layer.name, layer) for layer in model.layers])
#from keras import backend as K
#print(layer_dict)
#layer_name = 'conv2d_32'
#filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer
#
## build a loss function that maximizes the activation
## of the nth filter of the layer considered
#layer_output = layer_dict[layer_name].output
#loss = K.mean(layer_output[:, :, :, filter_index])
#
## compute the gradient of the input picture wrt this loss
#grads = K.gradients(loss, input_img)[0]
#
## normalization trick: we normalize the gradient
#grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
#
## this function returns the loss and grads given the input picture
#iterate = K.function([input_img], [loss, grads])
##Rearrange dimension so that we can plot the results as RGB img
#output_img= np.rollaxis(np.rollaxis(output_img,3,1),3,1)
#print(output_img.shape)
#
#fig=plt.figure(figsize=(8,8))
#for i in range(24):
#    ax= fig.add_subplot(6,6,i+1)
#    ax.imshow(output_img[0,:,:,i], cmap=matplotlib.cm.gray)# to see the first filter
#    plt.xticks(np.array([]))
#    plt.yticks(np.array([]))
#    plt.tight_layout()
#    plt



    #%%
    
    #visualising intermediate layers
output_layer = model.layers[5].output
output_fn= theano.function([model.layers[0].input], output_layer)
#
## input image
input_img =x_train[0:15,:,:,:]
print(input_img.shape)
plt.imshow(input_img[0,:,:,:],cmap='gray')
plt.imshow(input_img[0,0,:,:])
#%% input image
input_img = x_train[15,:,:,:]#[no of images:width:height:channel]
img = input_img[:,:,0] #[width:height:channel]
plt.imshow(img)
#%%

output_img = output_fn(input_img)
print(output_img.shape)

#Rearrange dimension so we can plot the result as RGB img
output_img= np.rollaxis(np.rollaxis(output_img,3,1),3,1)
print(output_img.shape)

# plot 
fig=plt.figure(figsize=(8,8))
for i in range(24):
    ax= fig.add_subplot(6,6,i+1)
    ax.imshow(output_img[0,:,:,i], cmap=matplotlib.cm.gray)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
    plt
#%%
model.summary()


model.get_config() # conf of every layer
model.layers[0].get_config() # conf of layer 0
model.layers[0].input_shape		# input shape of layer 0	
model.layers[0].output_shape	# output shape of layer 0		
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable # layer trainable or not

#%%
from keras.utils.vis_utils import plot_model
graph = plot_model(model,to_file='my_model.png', show_shapes=True)
#%%
model.save('pav_dis_cnn.hdf5')
#%%
from keras.models import load_model
model1 = load_model('pav_dis_cnn.hdf5')