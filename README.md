# iNES_lab_projects
This repository contains implimentation of vggnet(https://arxiv.org/pdf/1409.1556.pdf) by using tensorflow, and keras respectively in our own and tiny imagenet dataset. Moreover, it will tell us about transfer learning and fine tuning.
# 1)Vggnet experiments

1) Requirements:

 1.1) Tensorflow latest version.

 1.2) Keras latest version.

 1.3) python 3.6.

 1.4) Other machine learning and data handling libraries which I'm using in my codes.

# 2) Implementation with tensorflow on any custom dataset:

I tried to apply Vgg16 network upon any gerenal dataset by adding batch capabilities and using cross entropy loss in my code.
In the VggNet_tensorflow folder, we have all the necessary files to start training and test our model.

 2.1) Training:

  For running this code and start training on our dataset, we need to run training_vgg16.py file in terminal or your python
  simulator. In this code, I have added the ability to save the final trained model. In VggNet_tensorflow folder, we have data
  folder which contain the data but we can use any kind of data to train our model. The accuracy may be not so good because of 
  less data but we can try to use for large data as well.

 2.2) Testing: 

  For testing the saved model, please run new_model_test.py file and in this file, the code firstly load the already saved model 
  then load a simple test image and gave us prediction.

 2.3) Model:

  The VGG16_model.py contains the Network of VGGNet_16 and layerConstructor.py contains the defination of our layers of model.

# 3) Implementation with tensorflow on Tiny Imagenet dataset:

In this experiment, I tried to apply Vgg16_Net to tiny Imagenet dataset with different changes. The directory imagenet_tensor flow contain all the files except dataset so we can download the dataset from this following link: 

https://tiny-imagenet.herokuapp.com/

Please download the dataset from this link and placed in the same directory in which our code is present.

 3.1) Training:
 
  To start training our model on imagenet dataset we just need to run train.py file. This file have the ability to save model and
  test the model.
  
 3.2) Model:
 
  Model is in the file vgg16.py and our dataset has very low resolution therefore, I remove some layers from vggnet for example
  if we open the file and see 3 and 5 layers are not including. 
 
 3.3) Input_pipe.py
 
  This file contains loading images and labels from directories of imagenet and making quenes for training on gpu. The data
  preprocessing is done by cpu but trainig is done on gpu.
 
 3.4) Loss.py
 
  This file contains defination of three different loss function which we can be implemented according to requirement.
  I also define svm loss but it doesn't show any improvement.
  
# 4) Implementation with keras on our custom dataset by using transfer learning and fine tuning:

 In this experiment, I used a different plateform to train our model because Keras is easy to use and user fiendly language.And in the backend of it, we can use tensorflow or theano languages. But in my case, I,m using tensorflow in the backend of keras. The folder transfer_learning_vggnet16 contains the required files for running our code. 
 
 4.1) Training:
 
 To start traning the model please sun this file transfer_learning_vgg16_custom_data.py. This file contains the concept of  
 freezing the layers of network and start fine tuning our network. This is an easy example for learn about transfer learning.
 
 4.2) vgg16.py:
  
  It is the generalize network which downloads the weights for vggnet16 and then We can test and train a new network by using new 
  datasets.
  
# 5) Implementation with keras on Tiny Imagenet Dataset.

In this experiment, I implemented a simple cnn network in Tiny Imagenet Dataset by using keras. But if anyone want to change in network to vggnet then just remove the relu layers in it and add some other layers according to Vggnet Architecture. The folder Simple_CNN_imagenet contains the code file tiny_imagenet.py. And datset can be found in this following link:

https://tiny-imagenet.herokuapp.com/.

# Advisor and Collaborator list:

1.Advisor:

Name: Professor Jungsuk Kim,

Affiliation: Supervisor 

2.Co-Advisor:

Name: Dr. Peter Kim,

Affiliation: Co-Supervisor
