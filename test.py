# USAGE
# python predict_video.py --model model/activity.model --label-bin model/lb.pickle --input example_clips/lifting.mp4 --output output/lifting_128avg.avi --size 128

# import the necessary packages
from keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
from cv2 import cv
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
    help="path to  label binarizer")
ap.add_argument("-i", "--input", required=True,
    help="path to our input video")
ap.add_argument("-o", "--output", required=True,
    help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
    help="size of queue for averaging")
ap.add_argument("-of", "--optical-flow", required=False, type=bool,
    default=False, help="config that determines if the optical flow must be used")
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
QUIVER = 5

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        if args['optical-flow']:
            # create initial values for Optical Flow horizontal and vertical components
            U = cv.CreateMat(H, W, cv.CV_32FC1)
            V = cv.CreateMat(H, W, cv.CV_32FC1)
            
            # create a mask image for drawing purposes
            mask = np.zeros_like(frame)
            # create figure with mask background and remove the Axes
            fig, ax = plt.subplots()
            ax.imshow(mask)
            ax.set_axis_off()
            
            # save the actual frame and get the next
            old_frame = frame.copy()
            (grabbed, frame) = vs.read()
            if not grabbed:
                break
    
    # clone the output frame, then convert it from BGR to RGB
    # ordering
    output = frame.copy()        
    
    # if the optical optical flow must be used, convert the frame into a
    # cumulated optical flow image
    if args['optical-flow']:
        # convert the frames into opencv grayscale arrays
        old_gray_cv = cv.fromarray(cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY))
        frame_gray_cv = cv.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        # calculate optical flow
        cv.CalcOpticalFlowHS(old_gray_cv, frame_gray_cv, 0, U, V, 0.01, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # plot the optical flow
        Unp = np.asarray(U)
        Vnp = np.asarray(V)
        for i in range(0,len(Unp),QUIVER):
            for j in range(0,len(Vnp[0]),QUIVER):
                if abs(Vnp[i,j]) > 0.8 or abs(Unp[i,j]) > 0.8:
                    ax.arrow(j,i, Vnp[i,j], Unp[i,j], color='red')
        
        # now update the previous frame and previous points
        old_frame = frame.copy()
        
        # update the figure canvas with the plot and convert he to a numpy
        # array
        fig.canvas.draw()
        frame_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        frame_data = frame_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # make the frame be the optical flow plot
        frame = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
    
    # resize the frame to a fixed 224x224, and then
    # perform mean subtraction
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    frame -= mean

    # make predictions on the frame and then update the predictions
    # queue
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)

    # perform prediction averaging over the current history of
    # previous predictions
    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = lb.classes_[i]

    # draw the activity on the output frame
    text = "behavior: {}".format(label)
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
        1.25, (0, 255, 0), 5)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (W, H), True)

    # write the output frame to disk
    writer.write(output)

    # show the output image
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()