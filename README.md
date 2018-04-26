# iNES_lab_projects
This repository contains implimentation of vggnet by using tensorflow, and keras respectively in our own and tiny imagenet dataset. Moreover, it will tell us about transfer learning and fine tuning.
# 1)Vggnet experiments
1) Implementation with tensorflow on any custom dataset

I tried to apply Vgg16 network upon any gerenal dataset by adding batch capabilities and using cross entropy loss in my code.
In the VggNet_tensorflow folder, we have all the necessary files to start training and test our model.

1.1) Training

For running this code and start training on our dataset, we need to run training_vgg16.py file in terminal or your python simulator. In this code, I have added the ability to save the final trained model. In VggNet_tensorflow folder, we have data folder which contain the data but we can use any kind of data to train our model. The accuracy may be not so good because of less data but we can try to use for large data as well.
1.2) Testing 

For testing the saved model, please run new_model_test.py file and in this file, the code firstly load the already saved model then load a simple test image and gave us prediction.

1.3) Model

The VGG16_model.py contains the Network of VGGNet_16 and layerConstructor.py contains the defination of our layers of model.

2) Implementation with tensorflow on Tiny Imagenet dataset.

3) Implementation with keras on our custom dataset by using transfer learning and fine tuning.

4) Implementation with keras on Tiny Imagenet Dataset.
