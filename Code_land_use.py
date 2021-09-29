import tensorflow as tf 
from tensorflow import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau 

import random
import numpy as np
import seaborn as sns 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

import os
import glob
import zipfile
import pathlib
from pathlib import Path

# List name is list all class
def display_image(nrows, ncols, list_name):
    fig = plt.figure()
    fig.set_size_inches(ncols*4, nrows*4)

    # list path each class in train_set
    list_path_train_class = []
    for name in list_name:
      train_class_name = os.path.join(train, name)
      list_path_train_class.append(train_class_name)

    # Take 4 pic from every class
    list_img_path = []
    for pic in list_path_train_class:
      name_pic = glob.glob(f'{pic}/*')[:4]
      list_img_path.extend(name_pic)

    # Display image
    for index, img_path in enumerate(list_img_path):
      sp = plt.subplot(nrows, ncols, index+1)
      sp.axis('Off')
      img = mpimg.imread(img_path)
      plt.imshow(img)

    plt.show()

display_image(21,4, list_name_class)



"""#### Another way display image"""

def first_four_img_visualize(nrows,ncols,list_name):
    fig = plt.figure()
    fig.set_size_inches(ncols*4, nrows*4)

    list_path_train_class = []
    for name in list_name:
        train_class_name = os.path.join(train, name) 
        list_path_train_class.append(train_class_name)

    img_paths = []
    for path in list_path_train_class: 
        file_list = os.listdir(path)
        for index in range(0,4): 
            img_path = os.path.join(path, file_list[index])
            img_paths.append(img_path)

    for e, img_path in enumerate(img_paths):
      sp = plt.subplot(nrows, ncols, e + 1)
      sp.axis('Off') 
      img = mpimg.imread(img_path)
      plt.imshow(img)

    plt.show()

first_four_img_visualize(21,4,list_name_class)



"""### Function loss graph"""

def acc_loss_graph(history):
  #-----------------------------------------------------------
  # Retrieve a list of list results on training and test data
  # sets for each training epoch
  #-----------------------------------------------------------
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(acc))
  #------------------------------------------------
  # Plot training and validation accuracy per epoch
  #------------------------------------------------
  plt.plot(epochs, acc, 'b', label='Training accuracy')
  plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
  plt.legend()
  plt.title('Training and validation accuracy')

  plt.figure()
  #------------------------------------------------
  # Plot training and validation loss per epoch
  #------------------------------------------------
  plt.plot(epochs, loss, 'b', label='Training Loss')
  plt.plot(epochs, val_loss, 'r', label='Validation Loss')
  plt.title('Training and validation loss')
  plt.legend()

  plt.show()



###################
### Build Model ###
###################

### Configure image Data Generator

def generator_maker():

  train_datagen = ImageDataGenerator(rescale = 1./255, zoom_range = 0.2, horizontal_flip = True)
  val_test_datagen  = ImageDataGenerator(rescale=1./255)


  # train
  train_generator = train_datagen.flow_from_directory(train,
                                                target_size = (150,150),
                                                batch_size = 32,
                                                class_mode = 'categorical')

  # Validation
  val_generator = val_test_datagen.flow_from_directory(validation,
                                                  target_size = (150,150),
                                                  batch_size=32,
                                                  class_mode = 'categorical')

  # Test
  test_geneartor = val_test_datagen.flow_from_directory(test,
                                                    target_size = (150,150),
                                                    batch_size = 32,
                                                    class_mode = 'categorical')
  return train_generator, val_generator, test_geneartor

train_generator, val_generator, test_geneartor = generator_maker()

for batchs, labels in train_generator:
  print('Shape of batchs', batchs.shape)
  print('Shape of labels', batchs.shape)
  break

#------------------------------------------------#
      # Start with simple model, only Dense #
#------------------------------------------------#

model_simple = keras.Sequential([
        layers.Flatten(input_shape=(150,150,3)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(21, activation='softmax')                  
])

model_simple.summary()

model_simple.compile(optimizer='adam',
                     loss = 'categorical_crossentropy', # Why use categorical crossentropy instead of spares categorical crossentropy
                     metrics = ['accuracy'])

history_simple = model_simple.fit(train_generator,
                                  validation_data = val_generator,
                                  epochs = 10,
                                  verbose = 2)


#------------------------------------------------#
                # Go with CNN #
#------------------------------------------------#

"""
With dropout, early stop, learning rate decay, regularizers
"""

def model_CNN_maker():
  model = tf.keras.models.Sequential([
      Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
      MaxPooling2D(2,2),
      Conv2D(64, (3,3), activation='relu'),
      MaxPooling2D(2,2),
      Conv2D(128, (3,3), activation='relu'),
      MaxPooling2D(2,2),
      Conv2D(128, (3,3), activation='relu'),
      MaxPooling2D(2,2),
      Flatten(),
      Dense(512, activation='relu'),
      Dense(21, activation='softmax')
  ])
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',    
                metrics = ['accuracy'])
  return model

model_CNN = model_CNN_maker()
model_CNN.summary()

"""Train"""

history_CNN = model_CNN.fit(train_generator,
                            validation_data=val_generator,
                            epochs=10,
                            batch_size = 32,
                            verbose=1)

acc_loss_graph(history_CNN)

# Early stopping
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        min_delta=1e-2,
                                                        patience=10,
                                                        verbose=1,
                                                        restore_best_weights = True)

# Dropout
def model_CNN_dropout():
  model = tf.keras.models.Sequential([
      Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
      MaxPooling2D(2,2),
      layers.Dropout(0.2),
      Conv2D(64, (3,3), activation='relu'),
      MaxPooling2D(2,2),
      layers.Dropout(0.2),
      Conv2D(128, (3,3), activation='relu'),
      MaxPooling2D(2,2),
      layers.Dropout(0.2),
      Conv2D(128, (3,3), activation='relu'),
      MaxPooling2D(2,2),
      layers.Dropout(0.2),
      Flatten(),
      Dense(512, activation='relu'),
      Dense(21, activation='softmax')
  ])
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',    
                metrics = ['accuracy'])
  return model

model_CNN_drop_earlyStop = model_CNN_dropout()
model_CNN_drop_earlyStop.summary()

history_CNN_drop_earlyStop = model_CNN_drop_earlyStop.fit(train_generator,
                                                          validation_data=val_generator,
                                                          epochs=10,
                                                          verbose=2,
                                                          batch_size = 32,
                                                          callbacks = [earlystop])

acc_loss_graph(history_CNN_drop_earlyStop)


"""Try to tune learning rate"""

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2, min_lr=0.001)

# Fit again and add in learning rate
history_CNN_drop_earlyStop = model_CNN_drop_earlyStop.fit(train_generator,
                                                          validation_data=val_generator,
                                                          epochs=10,
                                                          batch_size=32,
                                                          verbose=1,
                                                          callbacks = [earlystop, reduce_lr])

acc_loss_graph(history_CNN_drop_earlyStop)

# Dropout and regularizers
def model_CNN_master():
  model = tf.keras.models.Sequential([
      Conv2D(32, (3,3),kernel_regularizer=regularizers.l2(0.0001), activation='relu', input_shape=(150,150,3)),
      MaxPooling2D(2,2),
      layers.Dropout(0.2),
      Conv2D(64, (3,3),kernel_regularizer=regularizers.l2(0.0001), activation='relu'),
      MaxPooling2D(2,2),
      layers.Dropout(0.2),
      Conv2D(128, (3,3),kernel_regularizer=regularizers.l2(0.0001), activation='relu'),
      MaxPooling2D(2,2),
      layers.Dropout(0.2),
      Conv2D(128, (3,3),kernel_regularizer=regularizers.l2(0.0001), activation='relu'),
      MaxPooling2D(2,2),
      layers.Dropout(0.2),
      Flatten(),
      Dense(512, activation='relu'),
      Dense(21, activation='softmax')
  ])
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',    
                metrics = ['accuracy'])
  return model

model_master = model_CNN_master()
model_master.summary()

history_model_master = model_master.fit(train_generator,
                                      validation_data=val_generator,
                                        epochs=20,
                                        batch_size=32,
                                        verbose=1,
                                        callbacks = [earlystop,reduce_lr])

acc_loss_graph(history_model_master)



#------------------------------------------------#
                # VGG 16 #
#------------------------------------------------#

from tensorflow.keras.applications.vgg16 import VGG16
model_vgg16 = VGG16(weights='imagenet', 
                    include_top=False, 
                    #classes=21, 
                    #classifier_activation='softmax', 
                    input_shape=[224,224,3])


for layer in model_vgg16.layers[:]:
  layer.trainable = False

for layer in model_vgg16.layers:
    print(layer, layer.trainable)

# Create the model
model_vgg16_import = keras.Sequential()

# Add the vgg16 convolutional base model
model_vgg16_import.add(model_vgg16)

# Add new layers
model_vgg16_import.add(Flatten())

model_vgg16_import.add(Dense(1024, activation='relu'))

model_vgg16_import.add(layers.Dropout(0.2))

model_vgg16_import.add(Dense(21, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model_vgg16_import.summary()

"""
Prepare image data generator
"""

def generator_maker_vgg16():

  train_datagen = ImageDataGenerator(rescale = 1./255, zoom_range = 0.2, horizontal_flip = True)
  val_test_datagen  = ImageDataGenerator(rescale=1./255)


  # train
  train_generator_vgg16 = train_datagen.flow_from_directory(train,
                                                target_size = (224,224),
                                                batch_size = 32,
                                                class_mode = 'categorical')

  # Validation
  val_generator_vgg16 = val_test_datagen.flow_from_directory(validation,
                                                  target_size = (224,224),
                                                  batch_size=32,
                                                  shuffle = False,
                                                  class_mode = 'categorical')

  # Test
  test_geneartor_vgg16 = val_test_datagen.flow_from_directory(test,
                                                    target_size = (224,224),
                                                    batch_size = 32,
                                                    class_mode = 'categorical')
  
  return train_generator_vgg16, val_generator_vgg16, test_geneartor_vgg16

train_generator_vgg16, val_generator_vgg16, test_geneartor_vgg16 = generator_maker_vgg16()



for batchs,label in val_generator_vgg16:
  print(label)
  break

base_dir = './images_train_test_val'

train =       os.path.join(base_dir, 'train')

train_list_test = os.listdir(train)

print(train_list_test)

train_list_test[6]



"""Fine tune with SGD and learning rate decay"""

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                              patience=5, 
                                              factor=0.2, 
                                              min_lr=0.001)

sgd = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum=0.9)

model_vgg16_import.compile(optimizer=sgd,
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])

history_model_vgg16_import = model_vgg16_import.fit(train_generator_vgg16,
                          validation_data=val_generator_vgg16,
                          steps_per_epoch = (train_generator_vgg16.samples/ train_generator_vgg16.batch_size),
                          epochs=20,
                          verbose=1,
                          callbacks = [earlystop,reduce_lr])

acc_loss_graph(history_model_vgg16_import)

history_model_vgg16_import = model_vgg16_import.fit(train_generator_vgg16,
                          validation_data=val_generator_vgg16,
                          steps_per_epoch = (train_generator_vgg16.samples/ train_generator_vgg16.batch_size),
                          epochs=20,
                          verbose=1,
                          callbacks = [earlystop,reduce_lr])

acc_loss_graph(history_model_vgg16_import)



"""Tune optimizer Adam"""

model_vgg16_import.compile(optimizer='adam',
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])

history_model_vgg16_import = model_vgg16_import.fit(train_generator_vgg16,
                          validation_data=val_generator_vgg16,
                          steps_per_epoch = (train_generator_vgg16.samples/ train_generator_vgg16.batch_size),
                          epochs=10,
                          verbose=1,
                          callbacks = [earlystop,reduce_lr])

acc_loss_graph(history_model_vgg16_import)


"""
Continue train

"""

model_vgg16_import.compile(optimizer='adam',
                          loss = 'categorical_crossentropy',
                          metrics = ['accuracy'])

history_model_vgg16_import = model_vgg16_import.fit(train_generator_vgg16,
                                                    validation_data=val_generator_vgg16,
                                                    steps_per_epoch = (train_generator_vgg16.samples/ train_generator_vgg16.batch_size),
                                                    epochs=5,
                                                    verbose=1,
                                                    callbacks = [earlystop,reduce_lr])

acc_loss_graph(history_model_vgg16_import)

loss, acc = model_vgg16_import.evaluate(test_geneartor_vgg16)


#------------------------------------------------#
                # Xception #
#------------------------------------------------#

"""
Prepare image data generator
"""

def generator_maker_Xception():

  train_datagen = ImageDataGenerator(zoom_range = 0.2, horizontal_flip = True)
  val_test_datagen  = ImageDataGenerator()


  # train
  train_generator_xception = train_datagen.flow_from_directory(train,
                                                target_size = (299,299),
                                                batch_size = 32,
                                                class_mode = 'categorical')

  # Validation
  val_generator_xception = val_test_datagen.flow_from_directory(validation,
                                                  target_size = (299,299),
                                                  batch_size=32,
                                                  shuffle = False,
                                                  class_mode = 'categorical')

  # Test
  test_geneartor_xception = val_test_datagen.flow_from_directory(test,
                                                    target_size = (299,299),
                                                    batch_size = 32,
                                                    class_mode = 'categorical')
  
  return train_generator_xception, val_generator_xception, test_geneartor_xception

train_generator_xception, val_generator_xception, test_geneartor_xception = generator_maker_Xception()


"""Build Xception"""

xception_model = keras.applications.Xception(weights='imagenet',
                                             input_shape = (299,299,3),
                                             include_top = False)
xception_model.trainable = False

"""The default input image size for Xception is 299x299 and scale input pixels between -1 and 1"""
def xception_maker():
  inputs = keras.Input(shape=(299,299,3))
  x = tf.keras.applications.xception.preprocess_input(inputs)
  x = xception_model(x, training=False)
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dropout(0.2)(x)
  outputs = layers.Dense(21, activation='softmax')(x)
  modelX = keras.Model(inputs, outputs)

  return modelX

modelX = xception_maker()
modelX.summary()

modelX.compile(optimizer='adam',
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])


"""New early stop, learning rate decay"""
callback = [EarlyStopping(monitor = 'val_loss', 
                          patience=5, 
                          restore_best_weights=True),

            ReduceLROnPlateau(monitor = 'val_loss',
                              patience = 5,
                              factor = 0.2,
                              min_lr = 1e-2)]

init_epochs = 40

history_xception = modelX.fit(train_generator_xception,
                              validation_data = val_generator_xception,
                              epochs = init_epochs,
                              callbacks = callback)

acc = history_xception.history['accuracy']
val_acc = history_xception.history['val_accuracy']

loss = history_xception.history['loss']
val_loss = history_xception.history['val_loss']


""" Fine tune """

print("Number of layers in the Xception model:", len(xception_model.layers))

# Unfreeze Xeception model
xception_model.trainable = True

# Fine-tune from this layer onward
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in xception_model.layers[:fine_tune_at]:
  layer.trainable =  False

modelX.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

modelX.summary()

checkpoint_filepath = 'Model_Xception_{epoch}.h5'

callbacks = [EarlyStopping(monitor='val_loss', 
                           patience=3, 
                           restore_best_weights=True, 
                           verbose=1),
             
             ModelCheckpoint(monitor='val_loss', 
                             filepath = checkpoint_filepath, 
                             save_weights_only=False, 
                             save_best_only=True, 
                             verbose=1),
             
             ReduceLROnPlateau(monitor = 'val_loss',
                              patience = 3,
                              factor = 0.2,
                              min_lr = 1e-2)]

init_epochs = 40
fine_tune_epochs = 40
total_epochs = init_epochs + fine_tune_epochs

history_xception_fine = modelX.fit(train_generator_xception, 
                         validation_data=val_generator_xception,
                         epochs=total_epochs,
                         initial_epoch=init_epochs,
                         callbacks=callbacks)

modelX.save('/content/gdrive/MyDrive/DATASET/Model_trained/model_xception.h5')

"""Let's check"""

acc += history_xception_fine.history['accuracy']
val_acc += history_xception_fine.history['val_accuracy']

loss += history_xception_fine.history['loss']
val_loss += history_xception_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training')
plt.plot(val_acc, label='Validation')
plt.xticks(range(total_epochs))
# plt.ylim([0.9, 1])
plt.plot([init_epochs,init_epochs],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training vs. Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training')
plt.plot(val_loss, label='Validation')
plt.xticks(range(total_epochs))
# plt.ylim([0, 0.5])
plt.plot([init_epochs,init_epochs],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training vs. Validation Loss')
plt.xlabel('epoch')
plt.show()

modelX.save('/content/gdrive/MyDrive/DATASET/Model_trained/model_xception_Land_use.h5')


"""Model evaluation"""
# Evaluate on test set
modelX.evaluate(test_geneartor_xception)

# Get class
val_generator_xception.class_indices

# Predict validation set
predictions = modelX.predict(val_generator_xception)

# Result predict is probability
predictions[0]

# Get class been predicted of validation set
pred_indices = np.argmax(predictions, axis=1)
pred_indices

# REAL CLASS OF VALIDATION SET (NOT PREDICT)
val_ground_truth = val_generator_xception.classes
val_ground_truth

# classification report
print(classification_report(
                            val_ground_truth,
                            pred_indices,
                            target_names = list(val_generator_xception.class_indices.keys())
                            ))

matrix = confusion_matrix(val_ground_truth, pred_indices)
target_names = list(val_generator_xception.class_indices.keys())
plt.figure(figsize=(15,10))

sns.heatmap(matrix,
            annot = True,
            yticklabels = target_names,
            cmap = "YlGnBu")


#------------------------------------------------#
                # Error Analysis #
#------------------------------------------------#

print(val_generator_xception.class_indices)

filenames = val_generator_xception.filenames
print(len(filenames))
print(filenames[500:505], '\n')

ground_truth = val_generator_xception.classes
print(len(ground_truth))
ground_truth

predictions = modelX.predict(val_generator_xception)

"""Code error analysis"""

# prediction_table is a dict with index, prediction, ground truth
prediction_table = {}
for index, val in enumerate(predictions):
    # get argmax index
    index_of_highest_probability = np.argmax(val)
    value_of_highest_probability = val[index_of_highest_probability]
    prediction_table[index] = [value_of_highest_probability, 
                               index_of_highest_probability, 
                               ground_truth[index]]
assert len(predictions) == len(ground_truth) == len(prediction_table)

def get_images_with_sorted_probabilities(prediction_table,
                                         get_highest_probability,
                                         label,
                                         number_of_items,
                                         only_false_predictions=False):
    sorted_prediction_table = [(k, prediction_table[k])
                               for k in sorted(prediction_table,
                                               key=prediction_table.get,
                                               reverse=get_highest_probability)]

    result = []
    for index, key in enumerate(sorted_prediction_table):
        image_index, [probability, predicted_index, gt] = key
        if predicted_index == label:
            if only_false_predictions == True:
                if predicted_index != gt:
                    result.append(
                        [image_index, [probability, predicted_index, gt]])
            else:
                result.append(
                    [image_index, [probability, predicted_index, gt]])
    return result[:number_of_items]


def plot_images(filenames, distances, message):
    images = []
    for filename in filenames:
        images.append(mpimg.imread(filename))
    plt.figure(figsize=(20, 15))
    columns = 5
    for i, image in enumerate(images):
        ax = plt.subplot(len(images) / columns + 1, columns, i + 1)
        ax.set_title("\n\n" + filenames[i].split("/")[-1] + "\n" +
                     "\nProbability: " +
                     str(float("{0:.2f}".format(distances[i]))))
        plt.suptitle(message, fontsize=20, fontweight='bold')
        plt.axis('off')
        plt.imshow(image)

        
def display(sorted_indices, message):
    similar_image_paths = []
    distances = []
    for name, value in sorted_indices:
        [probability, predicted_index, gt] = value
        similar_image_paths.append(str(val_dir) + '/' + filenames[name])
        distances.append(probability)
    plot_images(similar_image_paths, distances, message)

"""###### Let me display images, the Model with high confidence to predict"""

def high_confidence_of_class(list_name_class):
  
  for i in range(len(list_name_class)):
    message = f'\n HIGH Confidence of {list_name_class[i]}'
    Highest_confident = get_images_with_sorted_probabilities(prediction_table, True, i, 10, False)
    display(Highest_confident, message)

high_confidence_of_class(target_names)

"""###### Let me display images, the Model with low confidence to predict"""

def low_confidence_of_class(list_name_class):
  
  for i in range(len(list_name_class)):
    message = f"\n Low Confidence of {list_name_class[i]}"
    Lowest_confident = get_images_with_sorted_probabilities(prediction_table, False, i, 10, False)
    display(Lowest_confident, message)

low_confidence_of_class(target_names)

"""Look what images of these labels with the highest probability of containing another label"""

target_names = list(val_generator_xception.class_indices.keys())

def wrong_label(list_name_class):
  plt.figure(figsize=(15,10))
  for i in range(len(list_name_class)):
    message = f'\n incorrect {list_name_class[i]}'
    incorrect_images = get_images_with_sorted_probabilities(prediction_table, True, i, 10, True)
    display(incorrect_images, message)
  plt.show()

wrong_label(target_names)



#------------------------------------------------#
                # Predict real image #
#------------------------------------------------#

#OR CHOOSE A RANDOM PHOTO FROM PREDICTION FOLDER
TEST_SET = pathlib.Path()

def plot_predict(model, images_path): # Truyền vào link folder contain image
  # Random choice
  TEST_SET = pathlib.Path(images_path)
  img_path = random.choice(list(TEST_SET.glob('*')))
  
  # Convert image to arrray
  img        = image.load_img(img_path, target_size=(299, 299))
  img_array  = image.img_to_array(img)
  img_array  = np.expand_dims(img_array, axis=0)

  prediction = model.predict(img_array)

  for key, value in val_generator_xception.class_indices.items():
    if value == prediction[0].argmax():
      pred = key

  plt.figure(figsize=(10,10))
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.title('Prediction: ' + pred.upper())
  plt.axis('off')
  plt.grid(b=None)
  plt.show()


plot_predict(modelX, TEST_SET)

"""
Model Xception work good with image screenshot from google map.

Some case false predict because dataset not clear, shown in wrong label functions above
"""