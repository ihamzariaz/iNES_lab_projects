import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import np_utils
def training(train_x, train_y, valid_x=None, valid_y=None, format_size=[224, 224], batch_size=10, learn_rate=0.01, num_epochs=1, save_model=True, debug=False):
    
        
        
    #with tf.device('/gpu:0'):
    
    assert len(train_x.shape) == 4
    [num_images, img_height, img_width, num_channels] = train_x.shape
    num_classes = train_y.shape[-1]
    num_steps = int(np.ceil(num_images / float(batch_size)))

    # build the graph and define objective function
    graph = tf.Graph()
    with graph.as_default():
        # build graph
        train_maps_raw = tf.placeholder(tf.float32, [None, img_height, img_width, num_channels])
        train_maps = tf.image.resize_images(train_maps_raw, (format_size[0], format_size[1]))
        train_labels = tf.placeholder(tf.float32, [None, num_classes])
        # logits, parameters = vgg16(train_maps, num_classes)
        logits = vgg16(train_maps, num_classes, isTrain=True, keep_prob=0.6)

        # loss function
        #cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_maps,train_labels=y))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_labels)
        loss = tf.reduce_mean(cross_entropy)

        # optimizer with decayed learning rate
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learn_rate, global_step, num_steps*num_epochs, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # prediction for the training data
        train_prediction = tf.nn.softmax(logits)
    
    # train the graph
    #with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    with tf.Session(graph=graph) as session:
        # saver to save the trained model
        saver = tf.train.Saver()
        session.run(tf.initialize_all_variables())

        for epoch in range(num_epochs):
            for step in range(num_steps):
                offset = (step * batch_size) % (num_images - batch_size)
                batch_data = train_x[offset:(offset + batch_size), :, :, :]
                batch_labels = train_y[offset:(offset + batch_size), :]
                feed_dict = {train_maps_raw: batch_data, train_labels: batch_labels}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

                if debug:
                    if step % int(np.ceil(num_steps/2.0)) == 0:
                        print('Epoch %2d/%2d step %2d/%2d: ' % (epoch+1, num_epochs, step, num_steps))
                        print('\tBatch Loss = %.2f\t Accuracy = %.2f%%' % (l, accuracy(predictions, batch_labels)))
                        if valid_x is not None:
                            feed_dict = {train_maps_raw: valid_x, train_labels: valid_y}
                            l, predictions = session.run([loss, train_prediction], feed_dict=feed_dict)
                            print('\tValid Loss = %.2f\t Accuracy = %.2f%%' % (l, accuracy(predictions, valid_y)))

            print ('Epoch %2d/%2d:\n\tTrain Loss = %.2f\t Accuracy = %.2f%%' %
                   (epoch+1, num_epochs, l, accuracy(predictions, batch_labels)))
            if valid_x is not None and valid_y is not None:
                feed_dict = {train_maps_raw: valid_x, train_labels: valid_y}
                l, predictions = session.run([loss, train_prediction], feed_dict=feed_dict)
                print('\tValid Loss = %.2f\t Accuracy = %.2f%%' % (l, accuracy(predictions, valid_y)))

        # Save the variables to disk
        if save_model:
            save_path = saver.save(session, '/home/ines/Desktop/VggNet_tensorflow/model.ckpt')
            print('The model has been saved to ' + save_path)
        session.close()


# predictions is a 2-D matrix [num_images, num_classes]
# labels is a 2-D matrix like predictions
def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


if __name__ == '__main__':
#data loading for training
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
    tr_x, te_x, tr_y, te_y = train_test_split(x, y, test_size=0.2, random_state=2)
    # training on a subset to get a quick result
    print('Training ...')
    from VGG16_model import vgg16
    training(tr_x, tr_y, te_x, te_y, format_size=[224, 224], batch_size=64, learn_rate=1e-3, num_epochs=50)    




