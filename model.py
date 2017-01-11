
###  Here I imported all the packages I use ###

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation,Dropout,Lambda,ELU
from keras.models import Sequential
from keras.optimizers import Adam, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import pickle
import numpy as np
from PIL import Image
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import cv2
import os

###  Loading the training data set ###
# This outpuits three objects: features and labels store the steering angles and pictures. The offset lists assigns every picture a
# number. 0 if it is a center picture. 1 and -1 if it is a left/right camera picture. That way we can later alter the steering angle.

np.random.seed(123)
img_dir = "J:/datasets/data/data/" #Insert path to your training data here
telemetry_file = img_dir + "driving_log.csv"  #insert path to your telemetry file here
a =  pd.read_csv(telemetry_file,names=["center", "left", "right","steering", "throttle", "break", "speed"])
# We are only interested in pictuers of the car actually driving. Therefore remove small throttle values.
keep=a["throttle"]>0.1
a=a[keep].reset_index()
center_labels = np.array([f for f in a["steering"]])
left_labels = np.array([f+0 for f in a["steering"]])
right_labels = np.array([f+0 for f in a["steering"]])

# Next we remove some of the picturers with small steering angles.
l =np.where( (abs(center_labels)<=0.02))
print(l[0].shape)
fi=np.random.choice(np.array(l[0]),1900,replace=False)
filters=np.delete(np.arange(center_labels.shape[0]),fi)

# No output the labels, feature_files, and offsets
labels=np.hstack((center_labels[filters], left_labels[filters], right_labels[filters]))
feature_files = np.hstack((a["center"][filters],a["left"][filters].str.strip(),a["right"][filters].str.strip()))
repeats = int(labels.shape[0]/3)
offsets = [0 for i in range(repeats)] + [1 for i in range(repeats)] + [-1 for i in range(repeats)]

# This function resizes and image
def translate(image, x, y):
    #http://www.programcreek.com/python/example/87702/cv2.warpAffine
    columns, rows = 200,66
    return cv2.warpAffine(
        image,
        np.float32([[1, 0, x], [0, 1, y]]),
        (columns, rows)
    )
# This functions does several color based augmentations.
def color_image(image,op=0.7):
    # First we augment brightness and saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2]*=np.random.uniform(0.4,1.2) #Adjust the brightness
    hsv[:,:,1]*=np.random.uniform(0.5,1) #Adjust the saturation channel
    converted=cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img = np.copy(converted)
    # Then we overlay it with a random shadow
    s = np.random.choice(range(-60,100))
    skew = np.random.choice(range(-100,100)) # Skew of the shadow to the right or to the left
    tr = np.array([[0+s,110],[0+s+skew,0],[100+s+skew,0],[100+s,110]])
    op=np.random.uniform(0,op) # Opacitivy of the shadow
    overlay = np.copy(img)
    cv2.fillPoly(overlay, [tr], -1)
    cv2.addWeighted(overlay, op, img, 1 - op, 0, img)
    return img

# This function flips the image. Iff so, it will also reverse the original steering angle by multiplying it by -1
def flip_image(image, offset,angle,off_ang):
    if np.random.randint(2)==1: #flip a coin. If 1 then flip the image.
        image = np.fliplr(image)
        new_angle = (-1)*angle-offset*off_ang # Also reverse the steering angle
    else:
        new_angle = angle + offset*off_ang
    return image,new_angle

# We move images to simulate slopes. Vetical shift will by adjusted by a small angle.
def move_image(image, pvs,phs,off_sh=0.001):
    h_shift=np.random.choice(range(-phs,+phs))
    v_shift = np.random.choice(range(-pvs,+pvs))
    image = translate(image, -5+h_shift,-24+v_shift)
    return image,h_shift*off_sh

# We also Strech the images vertically to simulate slopes.
def stretch_image(img, stretch=10):
    s = np.random.uniform(-stretch,stretch)
    pts1 = np.float32([[0,0],[200,0],[200,66],[0,66]])
    pts2 = np.float32([[0,s],[200,s],[200,66],[0,66]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(200,66), borderMode=1)
    return dst

# This function does perspective warping and can simulate more aggressive curve angles.
def persp_warp(image,steering,warp):
    # https://github.com/vxy10/P3-BehaviorCloning
    rows = image.shape[0]
    cols = image.shape[1]
    war = np.random.uniform(-warp, warp)
    shift_start = [cols/2+war,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],shift_start])
    dsteering = war/(rows/2) * 360/(2*np.pi*25.0) / 6.0
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    steering +=dsteering
    return image,steering


# Here we load the validation data. This is recorded on track 1 and only using the center camera


def data_generator(offsets,fs,ls,bz=200,mode="disk",pvs=30, phs=20,debug=False,shuffle=True,off_ang=0.25,off_sh=0.001,warp=45):
    if shuffle==True: # For debugging purposes we can disable shuffling the dataset
        np.random.seed(123)
    features = np.copy(fs)
    labels   = np.copy(ls)
    offsets  = np.copy(offsets)
    print("shuffled the dataset")
    while 1:
        if shuffle==True:
        # Shuffle features, labels and offset data
            ind_shuffle = np.random.choice(range(features.shape[0]),features.shape[0],replace=False)
            features = features[ind_shuffle]
            labels = labels[ind_shuffle]
            offsets=offsets[ind_shuffle]
        for i in range(0,features.shape[0],bz): # Iterate over the dataset
            if (i+bz>features.shape[0]):
                    break #break if not enough samples
            # Create the batches for features and the labels
            batch_f = np.array([np.array(Image.open(img_dir+f).resize((210,105))) for f in features[i:i+bz]]).astype("float32")
            batch_l = np.copy(labels[i:i+bz])
            batch_o = np.copy(offsets[i:i+bz])
            new_features = []
            ### Pre-Processing and augmentation steps
            for image,im_ind in zip(batch_f, range(len(batch_f))):
                # Adjust brightness and add shadows
                img = color_image(image)

                # flip the image
                img, new_ang = flip_image(img,batch_o[im_ind],batch_l[im_ind],off_ang)
                batch_l[im_ind] = new_ang #adjust the steering angle

                # move the image
                img, ang_adapt = move_image(img, phs=phs,pvs=pvs,off_sh=off_sh)
                batch_l[im_ind]+=ang_adapt # adjust the steering angle

                # apply perspective warping
                img, new_ang = persp_warp(img,batch_l[im_ind],warp)
                img = stretch_image(img)
                batch_l[im_ind]=new_ang # adjust the steering angle
                new_features.append(img)

            new_features = np.array(new_features)
            #Normalize the data. The color adjustments can result in valus < 0 and/or < 255
            np.place(new_features, new_features>255,255)
            np.place(new_features, new_features<0,0)
            if debug==False:
                yield np.array(new_features), batch_l
            if debug==True:
                yield np.array(new_features), batch_l,features[i:i+bz]

# This is the model by NVIDIA proposed in https://arxiv.org/abs/1604.07316
# I adapted the lambda layer by comma.ai model.
def NVIDIAmodel():
    model = Sequential()
    ### Normalization layer
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=[66,200,3]))
    #Block1
    model.add(Conv2D(24,5,5,border_mode="valid",subsample=[2,2])) #24
    model.add(Activation('relu'))
    #Block2
    model.add(Conv2D(36,5,5,border_mode="valid",subsample=[2,2])) #36
    model.add(Activation('relu'))
    #Block3
    model.add(Conv2D(48,5,5,border_mode="valid",subsample=[2,2])) #48
    model.add(Activation('relu'))
    #Block4
    model.add(Conv2D(64,3,3,border_mode="valid",subsample=[1,1])) #64
    model.add(Activation('relu'))
    #Block5
    model.add(Conv2D(64,3,3,border_mode="valid",subsample=[1,1])) #64
    model.add(Activation('relu'))
    #Flatten
    model.add(Flatten())
    model.add(Dropout(0.6)) #dded some dropout for regularization
    # FC1
    model.add(Dense(100))
    model.add(Dropout(0.4)) #added some dropout for regularization
    model.add(Activation('relu'))
    # FC2
    model.add(Dense(50))
    model.add(Activation('relu'))
    # FC3
    model.add(Dense(10))
    model.add(Activation('relu'))
    # OUTPUT
    model.add(Dense(1))
    return model

if __name__ == "__main__":
    cur_dir=os.getcwd()
    optimizer = Adam(lr=0.0001) #decreased the learning rate for better convergence. This was trail and error.
    model = NVIDIAmodel()
    model.compile(loss="MSE", optimizer=optimizer)
    ###  Parameters ####
    nb_epoch = 15 #number of epochs
    off_ang = 0.09 #offset angle of the cameras
    off_sh = off_ang/100 #angle adjustments when moving the pictures horizontally
    warp=65 #Range of perspective warping
    pvs = 10 # Vertical movements
    phs = 15 # Horizontal movements
    bz = 200 # batch size
    samples_per_epoch=20000
    #checkpointer = ModelCheckpoint(filepath=cur_dir +"/model{epoch:02d}.h5", verbose=1, save_best_only=False)
    model.fit_generator(data_generator(fs=feature_files,ls=labels,offsets=offsets,bz=bz,mode="disk",pvs=pvs,phs=phs,off_ang=off_ang,off_sh=off_sh,warp=warp),samples_per_epoch = samples_per_epoch, nb_epoch = nb_epoch, verbose=1, callbacks=[], class_weight=None, nb_worker=1)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    #serialize weights to HDF5
    model.save_weights("model.h5")
    print("model exported")
