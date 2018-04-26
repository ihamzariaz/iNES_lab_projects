import numpy as np
import os
import time
from vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.layers import Dense, Input, Flatten
# Loading the training data
if __name__ == '__main__':
    PATH = os.getcwd()
    data_path = PATH + '/data'
    data_dir_list = os.listdir(data_path)
    img_data_list=[]
    for dataset in data_dir_list:
        img_list=os.listdir(data_path+'/'+ dataset)
        img_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
        for img in img_list:
            img_path = data_path + '/'+ dataset + '/'+ img
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            #x = x/255
            print('Input image shape:', x.shape)
            img_data_list.append(x)

    img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
    print (img_data.shape)
    img_data=np.rollaxis(img_data,1,0)
    print (img_data.shape)
    img_data=img_data[0]
    print (img_data.shape)
    num_classes = 4
    num_of_samples = img_data.shape[0]
    labels = np.ones((num_of_samples,),dtype='int64')

    labels[0:202]=0
    labels[202:404]=1
    labels[404:606]=2
    labels[606:]=3

    names = ['cats','dogs','horses','humans']

# convert class labels to on-hot encoding
    Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
    x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#########################################################################################
# Custom_vgg_model_1
#Training the classifier alone
#image_input = Input(shape=(224, 224, 3))
#
#model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')# if include_top= False there'll be no flatten layers
#model.summary()
#last_layer = model.get_layer('fc2').output
##x= Flatten(name='flatten')(last_layer)
#out = Dense(num_classes, activation='softmax', name='output')(last_layer) # we've 4 classes instead of 1000 so that last layer which is classifer layer will be 4.
#vgg16_model = Model(image_input, out)
#vgg16_model.summary()
## lets fix the network because initial feature will be same and we need to train only last layer of network
#for layer in vgg16_model.layers[:-1]:
#	layer.trainable = False
#
#vgg16_model.layers[3].trainable
#vgg16_model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])# loss fuction is crossentropy
#
#
#t=time.time()
##	t = now()
#hist = vgg16_model.fit(X_train, y_train, batch_size=64, epochs=50, verbose=1, validation_data=(X_test, y_test))# model has the batches capability
#print('Training time: %s' % (t - time.time()))
## to see total accuracy of model
#(loss, accuracy) = vgg16_model.evaluate(X_test, y_test, batch_size=64, verbose=1)
#print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


#############################################################################################

#Training the feature extraction also

    image_input = Input(shape=(224, 224, 3))

    model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

    model.summary()

    last_layer = model.get_layer('block5_pool').output
    x= Flatten(name='flatten')(last_layer)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    out = Dense(num_classes, activation='softmax', name='output')(x)
    vgg16_model2 = Model(image_input, out)
    vgg16_model2.summary()

# freeze all the layers except the dense layers
    for layer in vgg16_model2.layers[:-3]:
        layer.trainable = False
    vgg16_model2.summary()

    vgg16_model2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

    t=time.time()
#	t = now()
    hist = vgg16_model2.fit(X_train, y_train, batch_size=64, epochs=100, verbose=1, validation_data=(X_test, y_test))
    print('Training time: %s' % (t - time.time()))
    (loss, accuracy) = vgg16_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)

    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

# to test our model
    img_path = 'test7.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    print (x.shape)
    x = np.expand_dims(x, axis=0)
    print (x.shape)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    preds = vgg16_model2.predict(x)
    y_pred = np.argmax(preds)
    print(names[y_pred])
