# USAGE
# nohup python /home/almacedobr/TCCII/train.py -d /home/almacedobr/weizmann/saved_dataset_of/ -e 100 >/home/almacedobr/logs/train.log </dev/null 2>&1 &

import tensorflow as tf
# force enable eager execution
# tf.compat.v1.enable_eager_execution()

# set the matplotlib backend so figures can be saved in the background
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

# import the necessary packages

from resnet.ResNet18 import ResNet18
from tensorflow.keras.optimizers import SGD

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from scipy import interp
from imutils import paths
import numpy as np
import argparse
import pickle
import itertools
import os

# with tf.device('/cpu:0'):
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=False, default="behavior.model",
	help="path to output serialized model")
ap.add_argument("-l", "--label-bin", required=False, default="lb.pickle",
	help="path to output label binarizer")
ap.add_argument("-e", "--epochs", type=int, default=25,
	help="# of epochs to train our network for")
ap.add_argument("-b", "--batch", type=int, default=32,
	help="# of batch size to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-o", "--output", type=str, default="./output",
	help="path to output folder")
args = vars(ap.parse_args())

# process the image paths
def parse_image(filepath,labels):
  # extract the class label from the filename
  parts = tf.compat.v1.strings.split(filepath, os.path.sep, result_type="RaggedTensor")
  label = parts[-2] == labels

  # load the image, convert it to RGB channel ordering, and resize
  # it to be a fixed 224x224 pixels, ignoring aspect ratio
  image = tf.io.read_file(filepath)
  image = tf.image.decode_png(image, channels=3)
  image = tf.image.resize(image, [224, 224])

  # mean subtraction
  mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
  image -= mean
  return image, label

def get_labels(image,label):
    return label

def get_images(image,label):
    return image

# create the network output folder
print("[INFO] create the output folder")
if not os.path.exists(args["output"]) or not os.path.isdir(args["output"]):
    try:
        os.mkdir(args["output"])
    except OSError:
        print ("Failed to create output folder")
    else:
        print ("Output folder created successfully")

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
trainPaths = tf.data.Dataset.list_files(os.path.normpath(args['dataset'] + '/training' + '/*/*'))
testPaths = tf.data.Dataset.list_files(os.path.normpath(args['dataset'] + '/evaluation' + '/*/*'))
labels = os.listdir(os.path.normpath(args['dataset'] + '/training'))

# parsing train and evaluation data
print("[INFO] parse train and evaluation datasets...")
train = trainPaths.map(lambda filepath: parse_image(filepath, labels)).shuffle(buffer_size=2000)
test = testPaths.map(lambda filepath: parse_image(filepath, labels)).shuffle(buffer_size=2000)

trainSize = len(list(paths.list_images(os.path.normpath(args['dataset'] + '/training'))))
testSize = len(list(paths.list_images(os.path.normpath(args['dataset'] + '/evaluation'))))
labelsCount = np.unique(labels, axis=0).size

print("[INFO] number of classes...")
print(labelsCount)
print("[INFO] train dataset size...")
print(trainSize)
print("[INFO] evaluation dataset size...")
print(testSize)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# load the ResNet-50 network, with the images shapes and the labels quantity
model = ResNet18(input_shape=(224, 224, 3), classes=labelsCount)

# compile our model (this needs to be done after our setting our
# layers to being non-trainable)
print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training resnet cnn...")
H = model.fit(
	train.repeat().batch(args["batch"], drop_remainder=True),
	steps_per_epoch=trainSize // args["batch"],
	validation_data=test.repeat().batch(args["batch"]),
	validation_steps=testSize // args["batch"],
	epochs=args["epochs"])

# serialize the model to disk
print("[INFO] serializing network...")
model.save(os.path.normpath(os.path.join(args['output'], args["model"])))

# serialize the label binarizer to disk
f = open(os.path.normpath(os.path.join(args['output'], args["label_bin"])), "wb")
f.write(pickle.dumps(lb, protocol=2))
f.close()

# serialize the network training history to disk
f = open(os.path.normpath(os.path.join(args['output'], 'history')), "wb")
f.write(pickle.dumps(H.history))
f.close()

#testY = np.array([label.numpy() for label in test.map(get_labels).take((testSize // args["batch"]) * args["batch"])])
print("[INFO] construct data list...")
dataList = list((image.numpy(), label.numpy()) for image, label in test.take((testSize // args["batch"]) * args["batch"]))
print("[INFO] get labels...")
testY = np.array([o[1] for o in dataList])
print("[INFO] get images...")
testX = np.array([o[0] for o in dataList])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=args["batch"])
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

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
plt.plot(np.arange(0, N), H.history["loss"], label="trein_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="trein_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Loss e Acurácia de Treinamento no Dataset")
plt.xlabel("Época #")
plt.ylabel("Loss/Acurácia")
plt.legend(loc="lower left")
plt.savefig(os.path.normpath(os.path.join(args['output'], args["plot"])))

# plot all ROC curves
plt.figure(figsize = (10,7))
plt.plot(fpr["micro"], tpr["micro"],
         label='Curva ROC da média micro (área = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Curva ROC da média macro (área = {0:0.2f})'
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
plt.savefig(os.path.normpath(os.path.join(args['output'], 'roc.png')))

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

print("[INFO] training finished...")
