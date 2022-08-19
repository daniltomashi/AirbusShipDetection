# Project files in master branch

# AirbusShipDetection

Data was taken from this competition - https://www.kaggle.com/competitions/airbus-ship-detection/data

# About files
trained_models.ipynb - it's file with already trained models

coef_and_loss.py - file, where loss and metrics to segmentation model

classification_model.py - file with classification model

segmentation_model.py - file with segmentation model

train_models.py - file, where process of compiling and fitting data to our models

requirements.txt - what libraries and/or frameworks do we need

# About solution
First of all I decoded mask for images and then created from this data datasets with masks and images. My thought was: it will be more easier, faster and relaible to create two NN, one for classification(there is ship or no) and other for segmentation, so I did it. I have used convolution neural network with 3 layers for my classification model and U-Net NN for segmentation model.
