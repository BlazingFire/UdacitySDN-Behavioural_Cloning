
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Cropping2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Lambda, Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


def build_model():
    model = Sequential()

    # Preprocessimage
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

    # crop image to remove car hood and sky
    model.add(Cropping2D(cropping=((70,20),(0,0))))           
    #return model
# '''
    # The model is a simplified form of Nvidia model
    #layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
    model.add(Conv2D(24,5,5,subsample=(2,2)))
    model.add(Activation('elu'))

    #layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
    model.add(Conv2D(36,5,5,subsample=(2,2)))
    model.add(Activation('elu'))

    #layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
    model.add(Conv2D(48,5,5,subsample=(2,2)))
    model.add(Activation('elu'))

    #layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
    model.add(Conv2D(64,3,3))
    model.add(Activation('elu'))

    #layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
    model.add(Conv2D(64,3,3))
    
    model.add(Activation('elu'))
    model.add(Dropout(0.25))
    
    model.add(Flatten())

    #layer 6- fully connected layer 1
    model.add(Dense(128))
    model.add(Activation('elu'))

    #Adding a dropout layer to avoid overfitting.
    model.add(Dropout(0.25))

    #layer 7- fully connected layer 2
    model.add(Dense(64))
    model.add(Activation('elu'))


    #layer 8- fully connected layer 3
    model.add(Dense(10))
    model.add(Activation('elu'))

    #layer 9- fully connected layer 4
    model.add(Dense(1)) 
#     model.load_weights('model.h5')
    return model
# '''
def train_model(model,X_train,y_train):
    # using a smaller learning rate with Adam
    optimizer_ = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    # saving model after each poch
    model.compile(loss='mse',optimizer=optimizer_ , metrics=['accuracy'])
    filepath = "saved-model-low-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]
#     print(len(X_train))
    model.fit(X_train,y_train,epochs=5,verbose=1,callbacks=callbacks_list,validation_split=0.15,shuffle=True)
    
# this function is called if we are using generator version of code
def train_model_generator(model,train_generator,train_samples,validation_generator,validation_samples):
    optimizer_ = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='mse',optimizer=optimizer_ , metrics=['accuracy'])

    filepath = "saved-model-low-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]
    
    model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1,callbacks=callbacks_list,shuffle=True)

# to save the model after training finishes
def save_model(model):
    model.save('model.h5')