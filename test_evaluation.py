# -*- coding: utf-8 -*-
# USAGE
# nohup python /home/almacedobr/TCCII/test_evaluation.py -d /home/almacedobr/weizmann_test/saved_dataset_of/ -m /home/almacedobr/output_weizmann_of/behavior.model -l /home/almacedobr/output_weizmann_of/lb.pickle -his /home/almacedobr/output_weizmann_of/history -e 100 >/home/almacedobr/logs/evaluation.log </dev/null 2>&1 &
# set the matplotlib backend so figures can be saved in the background
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# import the necessary packages
import tensorflow as tf
from tensorflow.keras.models import load_model
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
ap.add_argument("-b", "--batch", type=int, default=32,
	help="# of batch size to train our network for")
ap.add_argument("-o", "--output", type=str, default="./output",
	help="path to output folder")
args = vars(ap.parse_args())

# create the network output folder
print("[INFO] create the output folder")
if not os.path.exists(args["output"]) or not os.path.isdir(args["output"]):
    try:
        os.mkdir(args["output"])
    except OSError:
        print ("Failed to create output folder")
    else:
        print ("Output folder created successfully")

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
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")


# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]
    
    # load the image, convert it to RGB channel ordering, and resize
    # it to be a fixed 224x224 pixels, ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224)).astype("float32")
    image -= mean
    
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data and labels to NumPy arrays and get the labels quantity
data = tf.convert_to_tensor(data)
labels = np.array(labels)
labelsCount = lb.classes_.shape[0]
print("[INFO] number of classes...")
print(labelsCount)

# perform one-hot encoding on the labels
testY = lb.transform(labels)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(data, batch_size=args['batch'])
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), labels=range(labelsCount), target_names=lb.classes_))

print("[INFO] predictions size...")
print(np.array(predictions).shape[0])

print("[INFO] construct ROC curve...")
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
plt.plot(np.arange(0, N), history["loss"], label="trein_loss", linestyle='solid')
plt.plot(np.arange(0, N), history["val_loss"], label="val_loss", linestyle='dotted')
plt.plot(np.arange(0, N), history["accuracy"], label="trein_acc", linestyle='dashed')
plt.plot(np.arange(0, N), history["val_accuracy"], label="val_acc", linestyle='dashdot')
plt.title("Loss e Acurácia de Treinamento no Dataset")
plt.xlabel("Época #")
plt.ylabel("Loss/Acurácia")
plt.legend(loc="lower left")
plt.savefig(os.path.join(args['output'], 'plot.png'))

# plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='Curva ROC da média micro (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Curva ROC da média macro (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

# get the color map cool with a labels range of colors tonality
cmap = plt.cm.get_cmap('cool', labelsCount)
for i in range(labelsCount):
    plt.plot(fpr[i], tpr[i], color=cmap(i), lw=2,
             label='Curva ROC da classe {0} (área = {1:0.2f})'
             ''.format(lb.classes_[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('Taxa de falso positivo')
plt.ylabel('Taxa de verdadeiro positivo')
plt.title('Curva ROC multi-classe')
plt.legend(loc='lower right')
plt.savefig(os.path.join(args['output'], 'roc.png'))

print("[INFO] construct confusion matrix...")
# construct the multi-class confusion matrix
confusion = confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))

# plot the confusion matrix
plt.figure(figsize = (10,7))
plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.cool)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(labelsCount)
plt.xticks(tick_marks, lb.classes_, rotation=45)
plt.yticks(tick_marks, lb.classes_)
thresh = confusion.max() / 2
for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
    plt.text(j, i, confusion[i,j], horizontalalignment='center', color='red' if confusion[i,j] > thresh else 'black')
plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predict Label')
plt.savefig(os.path.join(args['output'], 'confusion_matrix.png'))

print("[INFO] evaluation finished...")
