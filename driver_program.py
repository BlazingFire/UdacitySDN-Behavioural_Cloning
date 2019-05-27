import csv
import cv2
import numpy as np
from train import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)

# train_samples, validation_samples = train_test_split(lines,test_size=0.1,shuffle=True)
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
# train_generator = generator(train_samples, batch_size=32)
# validation_generator = generator(validation_samples, batch_size=32)

def without_generator():
    images=[]
    angles=[]
#     print(len(lines))
    for line in lines:
        for i in range(0,3):
#             print(line )
            name = './data/IMG/'+line[i].split('/')[-1]
#             print(name)
            center_image = cv2.imread(name)
            center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)

            center_angle = float(line[3])
            images.append(center_image)
            if(i==0):
                angles.append(center_angle)
            elif(i==1):
                angles.append(center_angle+0.2)
            elif(i==2):
                angles.append(center_angle-0.2)


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
# plt.imshow(X_train[0])
# plt.imsave('img.jpg',arr=X_train[0])

model = build_model()
train_model(model,X_train,y_train)
#train_model_generator(model,train_generator,train_samples,validation_generator,validation_samples)
save_model(model)