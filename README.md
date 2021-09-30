# Use CNN Model predict image from satellite

## I. Dataset
   - 21 Class
   - 500 images in single class
   - Used augmentation
   - Size 256 x 256
   <img width="753" alt="Screen Shot 2021-09-30 at 13 44 23" src="https://user-images.githubusercontent.com/86963378/135401176-686a2b42-dcf9-4164-8a0d-823467303d43.png">



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
  ### 1. Configure image Data Generator**
  - Batch_size = 32
  - Train set: 70%
  - Validation set: 20%
  - Test set: 10%

### 2. Build model

#### **a. Build model**
  <img width="328" alt="Screen Shot 2021-09-30 at 13 49 27" src="https://user-images.githubusercontent.com/86963378/135402041-018a6e47-a1e8-4888-96c1-6f69e1d1338c.png">
  
  - Accuracy on validation set: 80%
  - Loss of validation: 0.7
  - Quantitative epochs trained: 60 
  - Technique used:
      - Dropout
      - Max_pooling
      - Conv2D
      - Regukarizers L2
      - Reduce learning rate on plateau

#### **b. VGG16**
 <img width="328" alt="Screen Shot 2021-09-30 at 14 01 43" src="https://user-images.githubusercontent.com/86963378/135403385-9ff9861d-8b40-4cc0-92f1-b92bb2f521d9.png">
 
- Accuracy on validation set: 88%
- Loss of validation: 0.36
- Quantitative epochs trained: 55 
- Technique layers used:
   - Reduce learning rate on plateau
   - Steps_pre_epoch
   - Early stop
 
 #### **c. Transfer learning Xception**
  <img width="328" alt="Screen Shot 2021-09-30 at 14 04 46" src="https://user-images.githubusercontent.com/86963378/135403745-4529f397-0110-4e3d-a25e-a824f1b29385.png">

- Accuracy on validation set: 98,7%
- Loss of validation: 0.043
- Quantitative epochs trained: 57 
- Technique layers used:
   - Fine tune 32 layers in Xception
   - Tuning learning rate in optimizer
   - Reduce learning rate on plateau
- **Accuracy evaluated on test set: 98,57%**
- **Loss evaluated on test set: 0.0442**

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 299, 299, 3)]     0         
_________________________________________________________________
tf.math.truediv (TFOpLambda) (None, 299, 299, 3)       0         
_________________________________________________________________
tf.math.subtract (TFOpLambda (None, 299, 299, 3)       0         
_________________________________________________________________
xception (Functional)        (None, 10, 10, 2048)      20861480  
_________________________________________________________________
global_average_pooling2d (Gl (None, 2048)              0         
_________________________________________________________________
dropout (Dropout)            (None, 2048)              0         
_________________________________________________________________
dense (Dense)                (None, 21)                43029     
=================================================================
Total params: 20,904,509
Trainable params: 9,521,373
Non-trainable params: 11,383,136
_________________________________________________________________
```

#### **d. Error Analysis**

<img width="600" alt="Screen Shot 2021-09-30 at 14 14 48" src="https://user-images.githubusercontent.com/86963378/135405126-dbbd3457-f747-4e47-97f7-e1267c2e3beb.png">

Show some case don't clear to predict wrong or confuse

<img width="142" alt="Screen Shot 2021-09-30 at 14 17 26" src="https://user-images.githubusercontent.com/86963378/135405470-76944981-5647-4030-b81a-abf7fb687086.png">

Model predict golfcourse

<img width="600" alt="Screen Shot 2021-09-30 at 14 19 41" src="https://user-images.githubusercontent.com/86963378/135405798-46b891fe-ff6c-408d-a23c-55a2b581f17c.png">

Confuse to predict


  
#### **e. Predict real image**
Predict actually images

Harbor in Vung Tau beach

<img width="382" alt="Screen Shot 2021-09-30 at 00 09 17" src="https://user-images.githubusercontent.com/86963378/135406131-e47965c1-47ff-492f-849f-c45819aa6dc3.png">

Tan Son Nhat airport

<img width="382" alt="Screen Shot 2021-09-29 at 14 53 07" src="https://user-images.githubusercontent.com/86963378/135406159-ed1a6014-e6ab-4bb1-8ea3-d426985162ad.png">



