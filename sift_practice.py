#!/usr/bin/env python
'''
Attempting to write sift from scratch
This is some practive code for writing modules and a class for 
SIFT - spatially invariant feature tracking -
Author: Matthew Cruz
'''

import numpy as np
import cv2 
import cv2.cv as cv
from scipy import ndimage
from math import sqrt, atan

#get the captured image
#take that captured frame and compute the gaussian twice
#getthe difference of the gaussians and then convolve with the original image using each channgele is using RGB


def capture_frame():
	#sort through the first three video sources and try and capture
    import sys
    for i in range(3):
        print i
        cap = cv2.VideoCapture(i)
        if cap: break

    try: video_src = cap
    except: pass

    ret, frame  = video_src.read()
    return frame


class App:

	def __init__(self, video_src):
		self.s = 2
		sigma = 1.6
		self.kscale = 2**(1./self.s)
		frame  = np.copy(video_src)
		print video_src.shape
		self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		self.raw = np.copy(frame)
		self.temp = np.copy(self.gray).astype(float)
		self.sigmaX = sigma
		self.sigmaY = sigma
		self.GList = []
		self.DoGList = []
		self.LoGList = []
		self.resample = np.copy(self.gray)
		self.OctaveList = []

	def Blur(self):
		self.temp = cv2.GaussianBlur(self.temp,(9,9), (self.kscale)*self.sigmaX, (self.kscale)*self.sigmaY)
		return self.temp

	def GaussList(self, n):
		#create a lst of gaussian blurs with each blur separated by sigma*k
		for j in range(n):
			self.GList.append(self.Blur())		#run the blur function and append to a list
		return self.GList						#return the list of blurred imagefor each octace

	def shiftGrayscale(self, D):
		##shifts grayscales up if negative and scales by maximum
		#Can be applied because the next steps are comparisons of values and not absolute
		#no histogram equalization done
		if np.min(D)<0:
			D = -1*np.min(D) + D
			D =  255*D/float(np.max(D))
		else:
			pass
		#removed 255*... and put in if statement 
		return D

	def DoG(self, test):
		self.GaussList(self.s + 3)				#this number comes directly from the paper
		for i in range(len(self.GList)-1):
			D = np.subtract(self.GList[i+1], self.GList[i])		#D = (Gk - G)*I, where Gk is the gaussian kernel convolved with the Image
			D = self.shiftGrayscale(D)								#scale the subtraction for thresholding
			self.DoGList.append(D)

			if test:
				img_name = "DoG %d" % (i)
				cv2.imshow(img_name, D.astype(np.uint8))

		return self.DoGList

	def nextOctave(self):
		#need to downsample the original image by a factor (2)
		#then repeat the DoG
		#save the gaussian list and DoG list as a tuple into self.OctaveList
		#this takes the image two from top of stack GaussList and samples every second pixel in x and y direction
		#this pic represents twiced the initial sigma value
		#2**0.5 x 2**0.5 = 2 or 2*sigma
		self.DoG(False)
		self.resample = self.GList[2]
		##
		self.OctaveList.append((self.GList,self.DoGList))
		self.GList = []
		self.DoGList = []
		(r,c) = self.resample.shape
		resampled = cv2.resize(self.resample, (c/2,r/2)) #dont know why but row and column are switched?

		return resampled
		
	def contrast(self):
		pass
	def EliminatingEdgeResponse(self, keypoints, DList):
		#Computed for each keypoint
		#incomplete
		#DList is constructed as a list of (Dxx, Dyy, Dxy)
		for pt in keypoints:

			H = np.array([[Dxx, Dxy],[Dxy, Dyy]])
			Tr = Dxx + Dyy						#Dxx + Dyy = alpha + Beta
			Det = Dxx*Dyy-(Dxy)**2					#DxxDyy - (Dxy)^2 = alpha*Beta
			#check for Tr/Det < (r+1)**2/r where the threshold is determined by r =~10
			if Tr/Det < ((r+1)**2)/r:
				keypoints.remove(pt)

				
		return keypoints

	def Diff(self):
		#WIP
		#use the sobel operator to take a derivative of the DoG image for each coordinate
		Diff_List =[]
		for dog_img in self.DoGList:
			Dxx = cv2.Sobel(dog_img, -1, 2, 0)
			Dyy = cv2.Sobel(dog_img, -1, 0, 2)
			Dxy = cv2.Sobel(dog_img, -1, 1, 1)
			Diff_List.append((Dxx,Dyy,Dxy))

		return Diff_List

	def OrientationAssign(self):
		#WIP
		#L = G*I this is the self.Glist[i]
		L=self.Glist[i]
		m = sqrt((L[row+1,column] - L[row-1,column])**2 +(L[row, column+1]-L[row,column-1])**2)
		theta = 
 
	def Extrema(self):
		#pass
		#find extrema by comparing the DoGList

		keypoints = set()
		self.DoG(True)
		for i in range(1,3):
			keypointsH = set()
			keypointsL = set()
			#index through second and third arrays and compare agaisnt +1 and -1
			for row in range(1,len(self.DoGList[i])-1,3):
				for column in range(1,len(self.DoGList[i].T)-1,3):
					ref = self.DoGList[i][row][column]
					#if the values surrounding are all less or greater
					#threshold the top and bottom
					bottom = self.DoGList[i-1][row-1:row+2,column-1:column+2] 
					top = self.DoGList[i+1][row-1:row+2,column-1:column+2]
					middle = self.DoGList[i][row-1:row+2,column-1:column+2] 
					if np.max(bottom)<ref and np.max(top)<ref and np.max(middle)==ref:
						keypointsH.add((column,row))
					elif np.min(bottom)>ref and np.min(top)>ref and np.min(middle)==ref:
						keypointsL.add((column, row))

					column+=1

				row +=1
			keypoints = (keypoints | keypointsH | keypointsL)
			print len(keypoints)

		# print len(keypoints), ':', len(keypoints[0]), ':', len(keypoints[1])
		# print len(keypoints[0] & keypoints[1])
		for j in (keypoints):
			cv2.circle(self.raw,j, 3, (0,255,0),-1)
		cv2.imshow('keypoints', self.raw)
		cv2.waitKey(-1)
		#draw on the original raw image using colored dots





def Blur_test(video_src):
	test_img = App(video_src).Blur()
	cv2.imshow('test_img', test_img)

def Gauss_test():
	'''
	Test the Octave Method to see if multuple gaussian blurs are being 
	run and saved into an output list
	'''
	test_img_list  = App(video_src).GaussList(5)
	for n in range(len(test_img_list)):
		img_name = "Octave %d" % (n)
		cv2.imshow(img_name, test_img_lit[n])

def DoG_test(video_src):
	test = App(video_src) 
	test.DoG(True) #instantiated as test instead of doing a direct run like App(video_src).DoG(True)
	cv2.imshow('Gray Img', test.gray) # or can use App(video_src).gray
	cv2.waitKey(-1)

def nextOctave_test(video_src):
	resample = App(video_src).nextOctave()
	cv2.imshow('resample', resample.astype(np.uint8))
	cv2.waitKey(-1)


def main():

	video_src = capture_frame()
	#DoG_test(video_src)
	video_src=cv2.imread('jayz.jpg')
	#DoG_test(video_src)
	App(video_src).Extrema()
	#resample_test(video_src)

	'''
	ch = 0xFF & cv2.waitKey(0)
	if ch == 27:
		break
	'''
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()



