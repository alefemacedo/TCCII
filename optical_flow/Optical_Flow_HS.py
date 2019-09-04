# -*- coding: utf-8 -*-
"""
@author: Álefe Macedo
"""

import numpy as np
import cv2
from cv2 import cv
from matplotlib import pyplot as plt
import os
import sys

QUIVER = 5

def calcOpticalFlowHS(path, video_name, class_path_of, class_path_raw):
    success = False
    cap = cv2.VideoCapture(r'%s' % path)
        
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    if not ret:
        print ("Falha ao buscar o vídeo")
        sys.exit()

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_gray_cv = cv.fromarray(old_gray)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    
    # Create initial values for Optical Flow horizontal and vertical components
    U = cv.CreateMat(old_frame.shape[0], old_frame.shape[1], cv.CV_32FC1)
    V = cv.CreateMat(old_frame.shape[0], old_frame.shape[1], cv.CV_32FC1)
    
    # Create figure with mask background and remove the Axes
    fig, ax = plt.subplots(num=1)
    ax.imshow(mask)
    ax.set_axis_off()
    
    count = 1
    while(1):
        ret,frame = cap.read()
        
        if not ret:
            success = True
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_cv = cv.fromarray(frame_gray)
        
        # Calculate optical flow
        cv.CalcOpticalFlowHS(old_gray_cv, frame_gray_cv, 0, U, V, 0.01, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Plot the optical flow
        Unp = np.asarray(U)
        Vnp = np.asarray(V)
        
        for i in range(0,len(Unp),QUIVER):
            for j in range(0,len(Vnp[0]),QUIVER):
                if abs(Vnp[i,j]) > 0.8 or abs(Unp[i,j]) > 0.8:
                    ax.arrow(j,i, Vnp[i,j], Unp[i,j], color='red')
        
        plt.savefig(os.path.join(class_path_of, "%s%s%d" % (video_name, '_', count)))
        cv2.imwrite(os.path.join(class_path_raw, "%s%s%d%s" % (video_name, '_', count, '.png')), frame)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        old_gray_cv = cv.fromarray(old_gray)
        count += 1
        
    """cv2.destroyAllWindows()"""
    cap.release()
    return success
