"""
Created on Mon Aug  5 15:30:00 2019
@author: x.liang@westminster.ac.uk

#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
#https://github.com/anujshah1003/Transfer-Learning-in-keras---custom-data
"""
import numpy as np
import os
import time
from vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten,Dropout
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix,classification_report,roc_curve, auc,roc_auc_score
from itertools import cycle
from scipy import interp
from sklearn.utils.multiclass import unique_labels
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

############# Loading and pre-processing an image #############################
# =============================================================================
# img_path = 'images\elephant.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# print (x.shape)
# x = np.expand_dims(x, axis=0)
# print (x.shape)
# x = preprocess_input(x)
# print('Input image shape:', x.shape)
# plt.imshow(x[0])
# model = VGG16 (include_top ='True', weights='imagenet')
# prediction= model.predict(x)
# print('Predicted:', decode_predictions(prediction))
# =============================================================================

################# Loading BSL training data ###################################
# Loading the training data
PATH = os.getcwd()
# Define data path
#data_path = PATH + '/data'
data_path = PATH + '/data_v' #vertical stack over
#data_path = PATH + '/data_h'  #horizontal stack over
#data_path = PATH + '/data_test'
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
#		x = x/255
		#print('Input image shape:', x.shape)
		img_data_list.append(x)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
#print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
#print (img_data.shape)
img_data=img_data[0]
#print (img_data.shape)
#plt.imshow(img_data[0])



#################### Define the number of classes #############################
# dementia:0
# healthy:1 
num_classes = 2
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:79]=0  # dementia class 0
labels[79:]=1   # healthy class 1


names = ['dementia','healthy']

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

# Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)



###############################################################################
#                                                                             #
#   Custom_VGG16_model_1:  Training As a classifier alone                     #
#   Early Stopping is used for avoid Overfitting                              #
###############################################################################
image_input = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()

last_layer = model.get_layer('fc2').output
#x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output')(last_layer)
custom_vgg_model_1 = Model(image_input, out)
custom_vgg_model_1.summary()

for layer in custom_vgg_model_1.layers[:-1]:
	layer.trainable = False
#custom_vgg_model_1.layers[3].trainable



# Compile the model 
#custom_vgg_model_1.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

custom_vgg_model_1.compile(loss='categorical_crossentropy',optimizer='Adam' ,metrics=['accuracy'])



# Early Stoping
monitor = EarlyStopping(monitor='val_loss', min_delta=0,patience=5,verbose=1, mode='auto', baseline =None, restore_best_weights=True)


# Train the model 
t=time.time()
#hist = custom_vgg_model_1.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
hist = custom_vgg_model_1.fit(X_train, y_train, batch_size=1, epochs=1000, verbose=1, validation_data=(X_test, y_test), callbacks=[monitor])
print('Training time: %s' % ( time.time()-t))

# Get Early Stoping Epoch Number
epochs_num= monitor.stopped_epoch+1

# Print accuracy history records
print("\n History", hist.history)
train_accuracy= hist.history['acc']
print("[INFO] train_accuracy: {:.4f}%".format(train_accuracy[-6] * 100))
val_accuracy= hist.history['val_acc']
print("[INFO] val_accuracy: {:.4f}%".format(val_accuracy[-6] * 100))

# Evaluation the model 
(loss, val_accuracy) = custom_vgg_model_1.evaluate(X_test, y_test, batch_size=1, verbose=1)
print("[INFO] loss={:.4f}, Val_accuracy: {:.4f}%".format(loss,val_accuracy * 100))


# Save the model 
#custom_vgg_model_1.layers[-1].get_config()
custom_vgg_model_1.save_weights('C:\\Users\\user\\Documents\\Github_Clone\\deep-learning-models\\VGG16\\train_accuracy 87.5969% val_accuracy 93.9394%_19_11_2019\\my_vgg16_model_weights_classifier.h5')
custom_vgg_model_1.save('C:\\Users\\user\\Documents\\Github_Clone\\deep-learning-models\\VGG16\\train_accuracy 87.5969% val_accuracy 93.9394%_19_11_2019\\my_vgg16_model_classifier.h5')


###############################################################################
#                                                                             #
#  Custom_resnet_model_2: Fine Tune the Model                                 #
#  Dropout/Early Stopping is used for avoid overfitting                       #              
#                                                                             #
###############################################################################

image_input = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

model.summary()


last_layer = model.get_layer('fc2').output
x = Dense(128, activation='relu', name='fc3')(last_layer)
x = Dropout(0.4)(x)  # % of features dropped
#x = Dense(128, activation='relu', name='fc4')(x)
#x = Dropout(0.2)(x)
out = Dense(num_classes, activation='softmax', name='output')(x)
custom_vgg_model_2 = Model(image_input, out)
custom_vgg_model_2.summary()
# freeze all the layers except the dense layers
for layer in custom_vgg_model_2.layers[:-3]:
	layer.trainable = False

custom_vgg_model_2.summary()

plot_model(custom_vgg_model_2,to_file='C:\\Users\\user\\Documents\\Github_Clone\\deep-learning-models\\VGG16\\train_accuracy 87.5969%_val_accuracy 87.8788%_20_11_19_freeze\\XingVgg16Model.png')
SVG(model_to_dot(custom_vgg_model_2).create(prog='dot', format='svg'))

custom_vgg_model_2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# Early Stoping
monitor = EarlyStopping(monitor='val_loss', min_delta=0,patience=15,verbose=1, mode='auto', baseline =None, restore_best_weights=True)


t=time.time()
# Train the model 
#hist2 = custom_vgg_model_2.fit(X_train, y_train, batch_size=1, epochs=1000, verbose=1, validation_data=(X_test, y_test),callbacks=[monitor])
hist2 = custom_vgg_model_2.fit(X_train, y_train, batch_size=3, epochs=500, verbose=1, validation_data=(X_test, y_test), callbacks=[monitor])
print('Training time: %s' % (time.time()-t))

print("\n Model_VGG History", hist2.history)
train_accuracy2= hist2.history['acc']
print("[INFO] Model_VGG train_accuracy: {:.4f}%".format(train_accuracy2[-16] * 100))
val_accuracy2= hist2.history['val_acc']
print("[INFO] Model_VGG val_accuracy: {:.4f}%".format(val_accuracy2[-16] * 100))

# Evaluation the model 
(loss2, val_accuracy2) = custom_vgg_model_2.evaluate(X_test, y_test, batch_size=3, verbose=1)
print("[INFO] Model_VGG loss={:.4f}, Val_accuracy: {:.4f}%".format(loss2,val_accuracy2 * 100))


# Get Early Stoping Epoch Number
epochs_num= monitor.stopped_epoch+1


# Save the Model
custom_vgg_model_2.save_weights('C:\\Users\\user\\Documents\\Github_Clone\\deep-learning-models\\VGG16\\train_accuracy 87.5969%_val_accuracy 87.8788%_20_11_19\\my_vgg16_model_weights_classifier.h5')
custom_vgg_model_2.save('C:\\Users\\user\\Documents\\Github_Clone\\deep-learning-models\\VGG16\\train_accuracy 87.5969%_val_accuracy 87.8788%_20_11_19\\my_vgg16_model_classifier.h5')


###############################################################################
#                                                                             #
#         Model Performance Analysis                                          #
#                                                                             #
###############################################################################

######################   Plot Loss/Accuracy Results    ########################

# visualizing losses and accuracy
hist=hist2  #For custom_vgg_model_2 analysis
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(epochs_num)  # if Early stopping is used      
plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'],loc=0)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=0)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])



##### Plot Confusion Matrix, ROC curve, Classification Report  ################
preds_prob = custom_vgg_model_2.predict(X_test)
pred = np.argmax(preds_prob, axis=1)
true = np.argmax(y_test, axis=1)

# Plot Confusion Matrix - method 1
cm= confusion_matrix(true, pred)
print("confustion Matrix", cm)


#Print Classification Report 
target_names =['Dementia','Healthy']
p2=print(classification_report(true, pred,target_names=target_names))

# Plot Roc Method 1
dementia_pred_prob = custom_vgg_model_2.predict(X_test)[::,0]
healthy_pred_prob = custom_vgg_model_2.predict(X_test)[::,1]
fpr1, tpr1, _ = roc_curve(y_test[::,0],  dementia_pred_prob)
auc1 = roc_auc_score(y_test[::,0], dementia_pred_prob)
plt.plot(fpr1,tpr1,label="Dementia, auc="+str(auc1))
plt.legend(loc=0)
plt.show()

fpr2, tpr2, _ = roc_curve(y_test[::,1], healthy_pred_prob)
auc2 = roc_auc_score(y_test[::,1], healthy_pred_prob)
plt.plot(fpr2,tpr2,label="Healthy, auc="+str(auc2))
plt.legend(loc=0)
plt.show()


#########     Plot Confusion Matrix - method 2 ################################

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(true, pred, classes=np.array(target_names),
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(true, pred, classes=np.array(target_names), normalize=True,
                      title='Normalized confusion matrix')
plt.show()



#########      Plot ROC Curve- method 2   #####################################
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], preds_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), preds_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#Plot of a ROC curve for a specific class
plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Dementia')
plt.legend(loc="lower right")
plt.show()

#Plot ROC curves for the multiclass problem
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(target_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

################   Model Evaluation/Prediction  ###############################
#img_path = 'C:/Users/liangx/Documents/Github_Clone/deep-learning-models/data/dementia/1_left2d_big.png'
#img_path = 'C:/Users/liangx/Documents/Github_Clone/deep-learning-models/data/healthy/6_right2d_big.png'
#img_path ='C:/Users/liangx/Documents/Github_Clone/deep-learning-models/data_h/dementia/1_combine_2d_h.png'
#img_path ='C:/Users/user/Documents/Github_Clone/deep-learning-models/data_v/healthy/80_combine_2d_v.png'
#img_path ='C:/Users/user/Documents/Github_Clone/deep-learning-models/data/Testset_Xing/3_combine_2d_v.png'

#img_path ='C:/Users/user/Documents/Github_Clone/deep-learning-models/data_v/dementia/1_combine_2d_v.png'
#img_path ='C:/Users/user/Documents/Github_Clone/deep-learning-models/data/Testset_Xing/2_combine_2d_v.png'

img_path = 'C:/Users/user/Documents/Github_Clone/deep-learning-models/data/Testset_Xing/1_combine_2d_v.png'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = custom_vgg_model_1.predict(x)
print('Predicted:',preds)


###############################################################################
#                                                                             #
#         Load Pretrained Model for prediction                                #
#                                                                             #
###############################################################################

############    Load Our Pretrained Model      ################################
from keras.models import load_model
#pretrained_model_xing = load_model('C:/Users/user/Documents/Github_Clone/deep-learning-models/VGG16/81.8182%_28_10_2019_freeze/my_vgg16_model_classifier.h5')
pretrained_model_xing = load_model('C:/Users/user/Documents/Github_Clone/deep-learning-models/VGG16/train_accuracy 87.5969%_val_accuracy 87.8788%_20_11_19_freeze/my_vgg16_model_classifier.h5')

# Predict 
#img_path = 'C:/Users/user/Documents/Github_Clone/deep-learning-models/data/Testset_Xing/1_combine_2d_v.png'
img_path = 'E:\\Dunhill Medical Research Project\\Dunhill Project Data\\UCL_Tyron\\Validation_ProfWoll\\segmented\\0040_3\\0040_3_combine_2d_v.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = pretrained_model_xing.predict(x)
#preds = pretrained_model_xing.predict(X_test)
print('Predicted:',preds)


###############################################################################
#                                                                             #
#  Use Pre-trained Model as a Feature Extractor                               #
#                                                                             #
###############################################################################
pretrained_model_xing.summary()
feature_model = pretrained_model_xing
layer_name = 'fc2'
intermediate_layer_model = Model(inputs=feature_model.input,
                                 outputs=feature_model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(img_data)
#Use the model as an image feature extractor
print('Input image features:', intermediate_output)
