import csv
import cv2
import numpy as np
from train import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt
from keras.utils import plot_model

lines = []
#using local path of already provided data(after copying)
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)

# This is the generator part of the code but since the training dataset had only 40000 images(enough to fit in memory), it was faster to run code without generator
train_samples, validation_samples = train_test_split(lines,test_size=0.1,shuffle=True)
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(0,3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    center_image = cv2.imread(name)
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    correction = 0.18
                    if abs(center_angle) > 1:
                        correction = 0.25
                    if(i==0):
                        angles.append(center_angle)
                    elif(i==1):
                        angles.append(center_angle+correction)
                    elif(i==2):
                        angles.append(center_angle-correction)

                              
                    images.append(cv2.flip(center_image,1))
                    if(i==0):
                        angles.append(center_angle*-1)
                    elif(i==1):
                        angles.append((center_angle+correction)*-1)
                    elif(i==2):
                        angles.append((center_angle-correction)*-1)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Here we make X_train and y_train without using generator, storing them in arrays
def without_generator():
    images=[]
    angles=[]
#     print(len(lines))
    for line in lines:
        # for left, right and center images
        for i in range(0,3):
#             print(line )
            name = './data/IMG/'+line[i].split('/')[-1]
#             print(name)
            center_image = cv2.imread(name)
            # convert to RGB mode
            center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)

            # reading steering angle
            center_angle = float(line[3])
            images.append(center_image)
            correction = 0.18
            if abs(center_angle) > 1:
                correction = 0.25
            # center image
            if(i==0):
                angles.append(center_angle)
            #left image and corrections
            elif(i==1):
                angles.append(center_angle+correction)
            # right images and corrections
            elif(i==2):
                angles.append(center_angle - correction)

            # Flipped image for each type of image in provided dataset
            images.append(cv2.flip(center_image,1))
            if(i==0):
                angles.append(center_angle*-1)
            elif(i==1):
                angles.append((center_angle+0.2)*-1)
            elif( i==2):
                angles.append((center_angle-0.2)*-1)
                           # trim image to only see section with road
    X_train = np.array(images)
    y_train = np.array(angles)
#     print(len(X_train))
#         plt.imshow(X_train[0])
    return X_train,y_train
        
X_train, y_train= without_generator()

# the functions are imported from train.py
model = build_model()
#print(model.summary())
train_model(model,X_train,y_train)

# Can comment above line and uncomment below line to run code using generator
#train_model_generator(model,train_generator,train_samples,validation_generator,validation_samples)
save_model(model)

'''
# extra code
plt.imshow(X_train[0])
plt.imsave('first_image_uncropped.jpg',arr=X_train[0])


plot_model(model, to_file='model.png')
output = model.predict(X_train[0].reshape((1,160,320,3)))
output = np.reshape(output,(70,320,3))
print(output)
data=output
img2 = data - np.min(data);
img2 = img2 / np.max(img2);
plt.imshow(img2)
plt.imsave('first_image_cropped.png',arr=img2)


 '''