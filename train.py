# USAGE
# nohup python /home/almacedobr/TCCII/train.py -d /home/almacedobr/weizmann/saved_dataset_of/ -e 100 >/home/almacedobr/logs/train.log </dev/null 2>&1 &

# set the matplotlib backend so figures can be saved in the background
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from resnet.ResNet50 import ResNet50
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from imutils import paths
from scipy import interp
import numpy as np
import argparse
import pickle
import itertools
import cv2
import os

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
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
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
labelsCount = np.unique(labels, axis=0).size
print("[INFO] number of classes...")
print(labelsCount)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation/testing data augmentation object (which
# we'll be adding mean subtraction to)
valAug = ImageDataGenerator()

# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation
# objects
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

# load the ResNet-50 network, with the images shapes and the labels quantity
model = ResNet50(input_shape=(224, 224, 3), classes=labelsCount)

# compile our model (this needs to be done after our setting our
# layers to being non-trainable)
print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training resnet cnn...")
H = model.fit_generator(
	trainAug.flow(trainX, trainY, batch_size=32),
	steps_per_epoch=len(trainX) // 32,
	validation_data=valAug.flow(testX, testY),
	validation_steps=len(testX) // 32,
	epochs=args["epochs"])

# serialize the model to disk
print("[INFO] serializing network...")
model.save(os.path.join(args['output'], args["model"]))

# serialize the label binarizer to disk
f = open(os.path.join(args['output'], args["label_bin"]), "wb")
f.write(pickle.dumps(lb))
f.close()

# serialize the network training history to disk
f = open(os.path.join(args['output'], 'history'), "wb")
f.write(pickle.dumps(H.history))
f.close()

# serialize the dataset split
np.save(os.path.join(args['output'], 'trainX.npy'), trainX)
np.save(os.path.join(args['output'], 'trainY.npy'), trainY)
np.save(os.path.join(args['output'], 'testX.npy'), testX)
np.save(os.path.join(args['output'], 'testY.npy'), testY)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32) 
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

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
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(os.path.join(args['output'], args["plot"]))

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
plt.savefig(os.path.join(args['output'], 'roc.png'))

print("[INFO] construct confusion matrix...")
# construct the multi-class confusion matrix
confusion = confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))

# plot the confusion matrix
plt.figure()
plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.cool)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(labelsCount)
plt.xticks(tick_marks, lb.classes_, rotation=45)
plt.yticks(tick_marks, lb.classes_)
thresh = confusion.max() / 2
for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
    plt.text(j, i, confusion[i,j], horizontalalignment='center', color='white' if confusion[i,j] > thresh else 'black')
plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predict Label')
plt.savefig(os.path.join(args['output'], 'confusion_matrix.png'))

print("[INFO] training finished...")
