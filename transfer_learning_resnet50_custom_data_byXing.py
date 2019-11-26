"""
Created on Thur July 4 9:26:00 2019

@author: x.liang@westminster.ac.uk

# https://github.com/anujshah1003/Transfer-Learning-in-keras---custom-data
"""

import numpy as np
import os
import time
from resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Flatten
from imagenet_utils import preprocess_input, decode_predictions
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split ## it’s just an old way of doing split 
from sklearn.model_selection import train_test_split  ## similar way doing split
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
#get_ipython().run_line_magic('matplotlib', 'inline')

#############    Loading and pre-processing an image    #######################
img_path = 'images\elephant.jpg'

#Load the image using load_img() function specifying the target size.
img = image.load_img(img_path, target_size=(224, 224))

#Keras loads the image in PIL format (height, width) which shall be converted into NumPy format (height, width, channels) using image_to_array() function.
x = image.img_to_array(img)
print (x.shape)

#Then the input image shall be converted to a 4-dimensional Tensor (batchsize, height, width, channels) using NumPy’s expand_dims function.
x = np.expand_dims(x, axis=0)
print (x.shape)


#Normalizing the image
#Some models use images with values ranging from 0 to 1 or from -1 to +1 or “caffe” style. 
#The input_image is further to be normalized by subtracting the mean of the ImageNet data. 
#We don’t need to worry about internal details and we can use the preprocess_input() function from each model to normalize the image.
x = preprocess_input(x)    
print('Input image shape:', x.shape)
plt.imshow(x[0])

model =ResNet50(include_top ='True', weights='imagenet')
prediction= model.predict(x)
print('Predicted:', decode_predictions(prediction))



#################    Loading BSL training data    #############################
PATH = os.getcwd()
print("PATH", PATH)
# Define data path
#data_path = PATH + '/data' #separated 

data_path = PATH + '/data_v' #vertical stack over
#data_path = PATH + '/data_h'  #horizontal stack over
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('\n Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img 
		img = image.load_img(img_path, target_size=(224, 224)) 
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		print('Input image shape:', x.shape)
		img_data_list.append(x)


img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)

# Plot a image out 
#plt.imshow(np.uint8(img_data[0]))
plt.imshow(img_data[0])




####################    Define the number of classes ##########################
num_classes = 2
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')



labels[0:79]=0  # dementia class 0
labels[79:]=1   # healthy class 1

names = ['dementia','healthy']
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)
#print("Y", Y)

# Shuffle the dataset
xs,ys = shuffle(img_data,Y, random_state=2)  #if I use the same random_state with the same dataset, then I am always guaranteed to have the same shuffle
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2, random_state=2)




###############################################################################
#  * Custom_resnet_model_1                                                    #
#    Train the Model as a Classifier                                          #
#    Only the classifier layer (last layer) is trainable,                     # 
#    parameters of other layers are freezed.                                  #
#    Used with Smaller Datasets                                               #
#    Early Stopping is used for avoid Overfitting                             #                                                                #
###############################################################################
image_input = Input(shape=(224, 224, 3))

# Creat ResNet50 Model    
# "include_top= True" means include the final dense layers 
model_resnet = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')

# Print Model Layers Details/ Plot a Graph Layout  
model_resnet.summary()
plot_model(model,to_file='C:/Users/User/Documents/Github_Clone/deep-learning-models/ResNet50/ResNet50Model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))

# Get the last layer "avg_pool" out and from there to add/create your own network layers
last_layer = model_resnet.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='softmax2outputs')(x)
custom_resnet_model_1 = Model(inputs=image_input,outputs= out)

# Print custom_resnet_model_1 Layers Details/ Plot a Graph Layout 
custom_resnet_model_1.summary()
plot_model(custom_resnet_model_1,to_file='C:/Users/user/Documents/Github_Clone/deep-learning-models/ResNet50/train_accuracy 69.7674% val_accuracy_freeze/XingRestNet50Model_classifier.png')
SVG(model_to_dot(custom_resnet_model_1).create(prog='dot', format='svg'))


# Freeze the parameters of previous layers, except the last layer
for layer in custom_resnet_model_1.layers[:-1]:
	layer.trainable = False

#custom_resnet_model_1.layers[-1].trainable

#################    Compile the Model   ######################################
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
custom_resnet_model_1.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

#################     Train the Model  ########################################
monitor = EarlyStopping(monitor='val_loss', min_delta=0,patience=5,verbose=1, mode='auto', baseline =None, restore_best_weights=True)

t=time.time()
hist3 = custom_resnet_model_1.fit(X_train, y_train, batch_size=1, epochs=100, verbose=1, validation_data=(X_test, y_test),callbacks=[monitor])
print('Training time: %s' % (time.time()-t))

train_accuracy3= hist3.history['acc']
print("[INFO] Model_VGG train_accuracy: {:.4f}%".format(train_accuracy3[-6] * 100))

#################    Test the Model     #######################################
(loss3, accuracy3) = custom_resnet_model_1.evaluate(X_test, y_test, batch_size=1, verbose=3)

print("[INFO] loss={:.4f}, val_accuracy: {:.4f}%".format(loss3,accuracy3 * 100))

#################    Save the Model     #######################################
custom_resnet_model_1.save_weights('C:\\Users\\user\\Documents\\Github_Clone\\deep-learning-models\\ResNet50\\train_accuracy 69.7674% val_accuracy\\my_ResNet50_model_weights_classifier.h5')
custom_resnet_model_1.save('C:\\Users\\user\\Documents\\Github_Clone\\deep-learning-models\\ResNet50\\train_accuracy 69.7674% val_accuracy\\my_ResNet50_model_classifier.h5')



###############################################################################
# * Custom_resnet_model_2                                                     #
#   Fine Tune the Model                                                       #
#   Add on personalised dense layers (FC Layers)                              #
#   Only last 6 layers are trainable                                          #
#   Used with Larger Datasets                                                 #
###############################################################################

# Creat the Model    
model = ResNet50(weights='imagenet',include_top=False)
model.summary()

last_layer = model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
# a softmax layer for 4 classes
out = Dense(num_classes, activation='softmax',name='softmax2outputs')(x)

# this is the model we will train
custom_resnet_model_2 = Model(inputs=model.input, outputs=out)

custom_resnet_model_2.summary()

for layer in custom_resnet_model_2.layers[:-6]:
	layer.trainable = False

custom_resnet_model_2.layers[-1].trainable

# Compile the Model
custom_resnet_model_2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Train the Model
t=time.time()
hist = custom_resnet_model_2.fit(X_train, y_train, batch_size=3, epochs=30, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))

# Test the Model    
(loss, accuracy) = custom_resnet_model_2.evaluate(X_test, y_test, batch_size=3, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
custom_resnet_model_2.save_weights('C:/Users/liangx/Documents/Github_Clone/deep-learning-models/ResNet50/my_ResNet50_model_weights_finetune.h5')
custom_resnet_model_2.save('C:/Users/liangx/Documents/Github_Clone/deep-learning-models/ResNet50/my_ResNet50_models_finetune.h5')



###############################################################################
#                                                                             #
#  Model as a Feature Extractor                                               #
#                                                                             #
###############################################################################
model = ResNet50(weights='imagenet',include_top=False)
model.summary()

img_path = 'images\elephant.jpg'

#Load the image using load_img() function specifying the target size.
ima = image.load_img(img_path, target_size=(224, 224))

#Keras loads the image in PIL format (width, height) which shall be converted into NumPy format (height, width, channels) using image_to_array() function.
x = image.img_to_array(ima)

#Then the input image shall be converted to a 4-dimensional Tensor (batchsize, height, width, channels) using NumPy’s expand_dims function.
x = np.expand_dims(x, axis=0)

#Normalizing the image
x = preprocess_input(x)    

#Use the model as an image feature extractor
features=model.predict(x)
print('Input image features:', features)

 
######################   Plot Results   #######################################
import matplotlib.pyplot as plt
# visualizing losses and accuracy
hist=hist3
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(8) # epoch number

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])


##############  Model Evaluation/Prediction ################################### 

#img_path = 'C:/Users/liangx/Documents/Github_Clone/deep-learning-models/data/dementia/1_left2d_big.png'
#img_path = 'C:/Users/liangx/Documents/Github_Clone/deep-learning-models/data/healthy/6_right2d_big.png'
#img_path ='C:/Users/liangx/Documents/Github_Clone/deep-learning-models/data_h/dementia/1_combine_2d_h.png'
img_path ='C:/Users/user/Documents/Github_Clone/deep-learning-models/data_v/dementia/1_combine_2d_v.png'
img_path ='C:/Users/user/Documents/Github_Clone/deep-learning-models/data_v/healthy/81_combine_2d_v.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = custom_resnet_model_1.predict(x)
print('Predicted:',preds)



