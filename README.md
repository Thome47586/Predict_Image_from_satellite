# Use CNN Model predict image from sattellite

## I. Dataset
### Context
   The main aim of this project is the masking of regions where land is being used in satellite images obtained through Landsat which have a low spatial resolution.

### Content
   This dataset contains satellite images of 21 classes such as buildings, baseball fields, freeways, etc. The original size of the images is 256x256 pixels. Originally there were 100 images per class. After augmenting each image 4 times the size of each class was brought up to 500 images. This allows for making a more robust model.

### Acknowledgements
   The above dataset was obtained from the UC Merced Dataset
   Credits
   Dataset: Yi Yang and Shawn Newsam, "Bag-Of-Visual-Words and Spatial Extensions for Land-Use Classification," ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS), 2010.

link: https://www.kaggle.com/apollo2506/landuse-scene-classification

## II. Objective 
- Build CNN model from casual to complex have accuracy over 95% and loss under 0.1.
- Collect real images by screen shot on Google map and use model predict. 
- Understand reasons model predict false.

## III. Main content
### 1. Explore data
- Dataset have 21 class about how land is being used, whole images collect from satellite iobtained through Landsat which have a low spatial resolution and been done augmentation.
- With 500 images from single class, totally 21 class is 10.500 image, It's a great resource to build strong model and enough to split train, validation, test set

### 2. Build model
  #### **a. Configure image Data Generator**
  We need to prepare data set before training, from image diratory split train, validation, test set and scale data.
  Result:
  - Train set: 70%
  - Validation set: 20%
  - Test set: 10%

  #### **b. Try to build model**
  In this lab, I trial and error from simple to complex Models to build myself. During which layers used: Dense, Max pooling to keep highlight elements
  As a result, the accuracy increases but an overfit signal appears, I limit the overfit by using dropout to randomly set some activations to 0. But this situation is not feasible I combine more regularizers l2
  and to reduce the loss I use to reduce the Learning rate

  As mentioned above, I trial and error, using knowledge I have learned to apply in practice to see how effective they are and what is different.
  To make it easier for you to imagine, I have omitted the code.

  After trained, I have some conclusions as follows:
  - Low-resolution satellite images make it difficult to train models
  - The model's built me, weights are retrained from the beginning, so it takes a long time in the fine-tuning get objective

  After testing and reading the document, I decided to switch to transfer learning VGG16 

  #### **c. Transfer learning VGG16**
  With Model VGG16 I trained more than 55 epochs, fine-tune many times and achieving acc 88% and loss 0.36. 
  During this time I noticed another model smaller than VGG16 and more efficient, this is Xception. 

  While I couldn't think of a way to improve VGG16 yet, I try to Xception

  #### **d. Transfer learning Xception**
  Over expectedly, in the first 40 epochs, the accuracy increased to 96% and the loss decreased to 0.1. I'm really happy with a positive response from Xception.
  Nextstep,  I opened 32 layers to fine-tune, add the callbacks I wanted and continue to train more 15 epochs, model get over 98% and loss dropped to 0.046.

  This is a great result during training.

  All these results are evaluated on the training and validation set. To ensure we need to evaluate it on test set hidden before and results are absolutely amazing, it's same as on the validation set.

  To evaluate it more objectively, I represented it with a classification report where the classes were been predicted incorrectly very clear. 
  But it's difficult to understand why it's wrong, so let's move error analysis 

  #### **e. Error Analysis**

  In Error Analysis I will display which class model high confidence low confidence and wrong label.
  In this section, we focus on image predicted wrong by model and find out solution

  #### **f. Predict real image**

  In the objective above, I collected images from Google map and test on model.
  With high accuracy and low loss, model predicts exactly even image low spatial resolution than dataset

