#!/usr/bin/env python
## COrrect usage of this file is as follows:
## on the command line ./contrast_hist.py filename.jpg
#takes filename as input on command line

from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv2
import cv2.cv as cv
from cv_bridge import CvBridge, CvBridgeError
import sys


def read_img(argv):
    img = cv2.imread(argv, 0)
    cv2.imwrite('original.png',img)
    cv2.imshow('raw_img', img)
    return img   

def histogram(argv):
	filename = argv
	img = read_img(filename)
	img_temp = img*1

	#create the bins for the histogram function
	hbins = np.linspace(0,255,256)

	#create the histogram using the numpy histogram function
	hist0, bin_edges0 = np.histogram(img,bins=256,density = False)
	hist, bin_edges = np.histogram(img,bins=256,density = True)

	###Loop histogram and get cumulative density* scale (255)"###

	pixtotal= len(img)*len(img[0])
   	freqtot = 0
   	final = np.zeros((256))
   	for k in range(0, len(hist0)):
   		freqtot += hist0[k]
   		final[k] = freqtot*255/(pixtotal)

   	for r in range(0,len(img)):                
	    for c in range(0,len(img[0])):         
    		img_temp[r,c]= final[img[r,c]]

	hist_temp, bin_edges_temp = np.histogram(img_temp,bins=256,density = True)
	#cv2.imshow('blah', img_temp)
	cv2.imwrite('contrast.png', img_temp)
	cv2.imshow('contrast enhanced', img_temp)

	plt.plot(hbins, final/255)
	plt.ylabel('Normal Cumulative Density')
	plt.xlabel('Pixel Value')
	plt.title('Normalized Cumulative Distribution')
	plt.figure()
	plt.plot(hbins, hist)
	plt.ylabel('Normalized Frequency Distribution')
	plt.xlabel('Pixel Value')
	plt.title('Normalized Histrogram of Pixel Values')
	plt.figure()
	plt.plot(hbins, hist_temp)
	plt.ylabel('Normalized Frequency Distribution')
	plt.xlabel('Pixel Value')
	plt.title('Post Processing: Normalized Histrogram of Pixel Values')
	plt.show()
	cv2.waitKey(-1)

	


histogram(sys.argv[1])
