#!/usr/bin/env python
# coding: utf-8

# # Note for 02_train_rgb_finetuning.py

# ###  Import libaries

# In[1]:


get_ipython().system('pip3 install tensorflow')
get_ipython().system('pip3 install keras')
get_ipython().system('pip install --upgrade numpy')


# In[17]:


import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16 as VGG
from keras.applications.densenet import DenseNet201 as DenseNet
from keras.optimizers import SGD
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from image_functions import preprocessing_image_rgb


# ### define path to training and validation data

# In[26]:


get_ipython().system('pwd')


# In[31]:


import re
class RunLocationError(Exception):
    pass



_pattern = re.compile("CNN-Sentinel$")
_match = re.search(_pattern, PROJECT_ROOT)

try:
    if not _match:
        raise RunLocationError
except RunLocationError:
    print(PROJECT_ROOT)
    print("ERROR: Run this from root of the project repository which should end be called 'CNN-Sentinel'")


# In[60]:


# variables
path_to_split_datasets = "~/code/CNN-Sentinel/data/ml_wrokflow/EuroSATallBandsTIF"
use_vgg = True
batch_size = 64

# contruct path
path_to_home = os.path.expanduser("~")
path_to_split_datasets = path_to_split_datasets.replace("~", path_to_home)
path_to_train = os.path.join(path_to_split_datasets, "train")
path_to_validation = os.path.join(path_to_split_datasets, "validation")

print (path_to_split_datasets, path_to_train, path_to_validation)

for path in (path_to_split_datasets, path_to_train, path_to_validation):
    cmd = f"tree {path}"
    get_ipython().system(cmd)


# ![tree](images_for_notebook/tree_files.png "file_tree")

# ### determine number of classes from data

# In[61]:


# get number of classes
sub_dirs = [sub_dir for sub_dir in os.listdir(path_to_train)
            if os.path.isdir(os.path.join(path_to_train, sub_dir))]
num_classes = len(sub_dirs)
assert(num_classes==10)
print (num_classes)


# ## Transfer-learning 

# ![vgg16](images_for_notebook/vgg16.png "Original VGG")

# ### 1. Pretrained network model without top layers

# ![vgg16_no_top](images_for_notebook/vgg16_no_top.png "VGG no top")

# In[62]:


# parameters for CNN
if use_vgg:
    base_model = VGG(include_top=False,
                     weights='imagenet',
                     input_shape=(64, 64, 3))
else:
    base_model = DenseNet(include_top=False,
                          weights='imagenet',
                          input_shape=(64, 64, 3))


# ### 2. define new top layers

# ![vgg16_sentinel_rgb](images_for_notebook/vgg16_sentinel_rgb.png "VGG RGB Sentinel")

# In[63]:


# add a global spatial average pooling layer
top_model = base_model.output
top_model = GlobalAveragePooling2D()(top_model)
# or just flatten the layers
#    top_model = Flatten()(top_model)
# let's add a fully-connected layer
if use_vgg:
    # only in VGG19 a fully connected nn is added for classfication
    # DenseNet tends to overfitting if using additionally dense layers
    top_model = Dense(2048, activation='relu')(top_model)
    top_model = Dense(2048, activation='relu')(top_model)
# and a logistic layer
predictions = Dense(num_classes, activation='softmax')(top_model)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# print network structure
model.summary()


# ### 3. define data augmentation

# In[64]:


# defining ImageDataGenerators
# ... initialization for training
train_datagen = ImageDataGenerator(fill_mode="reflect",
                                   rotation_range=45,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   preprocessing_function=preprocessing_image_rgb)
# ... initialization for validation
test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_image_rgb)
# ... definition for training
train_generator = train_datagen.flow_from_directory(path_to_train,
                                                    target_size=(64, 64),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
# just for information
class_indices = train_generator.class_indices
print(class_indices)

# ... definition for validation
validation_generator = test_datagen.flow_from_directory(path_to_validation,
                                                        target_size=(64, 64),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')


# ### 4. define callbacks

# In[65]:


# generate callback to save best model w.r.t val_categorical_accuracy
if use_vgg:
    file_name = "vgg"
else:
    file_name = "dense"

checkpointer = ModelCheckpoint("../data/models/" + file_name +
                               "_rgb_transfer_init." +
                               "{epoch:02d}-{val_categorical_accuracy:.3f}." +
                               "hdf5",
                               monitor='val_categorical_accuracy',
                               verbose=1,
                               save_best_only=True,
                               mode='max')

earlystopper = EarlyStopping(monitor='val_categorical_accuracy',
                             patience=10,
                             mode='max',
                             restore_best_weights=True)

tensorboard = TensorBoard(log_dir='./logs', write_graph=True, write_grads=True,
                          write_images=True, update_freq='epoch')


# ![tensorflow](images_for_notebook/tensorflow.png "VGG RGB Sentinel")

# ### 5. set base layers non trainable 

# ![vgg16_rgb_init](images_for_notebook/vgg16_rgb_init.png "VGG RGB Sentinel")

# In[66]:


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adadelta', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


# ### 6. fit model (train new top layers)

# In[70]:


history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=5,
                              callbacks=[checkpointer, earlystopper,
                                         tensorboard],
                              validation_data=validation_generator,
                              validation_steps=500)
initial_epoch = len(history.history['loss'])


# ### 7. set (some) base layers trainable

# ![vgg16_rgb_finetune](images_for_notebook/vgg16_rgb_finetune.png)

# In[ ]:


# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
names = []
for i, layer in enumerate(model.layers):
    print([i, layer.name, layer.trainable])

if use_vgg:
    # we will freaze the first convolutional block and train all
    # remaining blocks, including top layers.
    for layer in model.layers[:4]:
        layer.trainable = False
    for layer in model.layers[4:]:
        layer.trainable = True
else:
    for layer in model.layers[:7]:
        layer.trainable = False
    for layer in model.layers[7:]:
        layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


# ### 8. fit model (fine-tune base and top layers)

# In[11]:


# generate callback to save best model w.r.t val_categorical_accuracy
if use_vgg:
    file_name = "vgg"
else:
    file_name = "dense"
checkpointer = ModelCheckpoint("../data/models/" + file_name +
                               "_rgb_transfer_final." +
                               "{epoch:02d}-{val_categorical_accuracy:.3f}" +
                               ".hdf5",
                               monitor='val_categorical_accuracy',
                               verbose=1,
                               save_best_only=True,
                               mode='max')
earlystopper = EarlyStopping(monitor='val_categorical_accuracy',
                             patience=50,
                             mode='max')
model.fit_generator(train_generator,
                    steps_per_epoch=100,
                    epochs=initial_epoch+5,
                    callbacks=[checkpointer, earlystopper, tensorboard],
                    validation_data=validation_generator,
                    validation_steps=500,
                    initial_epoch=initial_epoch)

