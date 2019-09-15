# -*- coding: utf-8 -*-
# USAGE
# python predict_video.py --model model/activity.model --label-bin model/lb.pickle --input example_clips/lifting.mp4 --output output/lifting_128avg.avi --size 128

# import the necessary packages
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
from scipy import interp
import numpy as np
import argparse
import os
import pickle
import itertools
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
    help="path to  label binarizer")
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-his", "--history", required=True,
	help="path to network training history")
ap.add_argument("-e", "--epochs", type=int, default=25,
	help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="./output/plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
print("[INFO] loading model, label binarizer and training history...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())
history = pickle.loads(open(args["history"], "rb").read())

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# load the image, convert it to RGB channel ordering, and resize
	# it to be a fixed 224x224 pixels, ignoring aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

# convert the data and labels to NumPy arrays and get the labels quantity
data = np.array(data)
labels = np.array(labels)
labelsCount = lb.classes_.shape
print("[INFO] number of classes...")
print(labelsCount)

# perform one-hot encoding on the labels
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32) 
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# construct the roc curves and calc the aucs to min and macro averages
# as well as for each class

# compute ROC curve and ROC AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(labelsCount):
    fpr[i], tpr[i], _ = roc_curve(testY[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# compute micro-average ROC curve and ROC AUC
fpr["micro"], tpr["micro"], _ = roc_curve(testY.ravel(), predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
# compute macro-average ROC curve and ROC AUC

# first aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(labelsCount)]))

# then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(labelsCount):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# finally average it and compute AUC
mean_tpr /= labelsCount
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# plot the training loss and accuracy
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history["acc"], label="train_acc")
plt.plot(np.arange(0, N), history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

# get the color map cool with a labels range of colors tonality
cmap = plt.cm.get_cmap('cool', labelsCount)
for i in range(labelsCount):
    plt.plot(fpr[i], tpr[i], color=cmap(i), lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(lb.classes_[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve multi-class')
plt.legend(loc='lower right')
plt.savefig('./output/roc.png')

# construct the multi-class confusion matrix
confusion = confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))

# plot the confusion matrix
plt.figure()
plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.cool)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(labelsCount)
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
thresh = confusion.max() / 2
for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
    plt.text(i, j, confusion[i,j], horizontalalignment='center', color='white' if confusion[i,j] > thresh else 'black')
plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predict Label')