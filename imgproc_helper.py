#!/usr/bin/env python
'''
This file contains low level functions written from scratch to be used as helper functions

Author: Matthew Cruz
'''

import numpy as np
import cv2 
import cv2.cv as cv
from math import pi, exp, cos, sin
#from scipy import ndimage
#from math import sqrt, atan

def walk_folder(walk_dir):
	#place files into a folder and provide that directory as 'walk_dir'
	#should return a list of image files
    walk_dir  = '/home/mtcruz/Coursework/EECS332_Digital_Image/HW4/image_bank'
    for root_dir, subdirs, files in os.walk(walk_dir, topdown=True):
        numOfFiles = len(files)

    return files

def histogram(img):
	#requires image to be in HSV
	img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hs_map = np.zeros((250,250))

	for r in range(0,len(img)):                
	    for c in range(0,len(img[0])):  
	    	h=img[r,c,0]      
	    	s= img[r,c,1]
	    	hs_map[h,s] +=1

	return hs_map

class Helper:

	def __init__(self):
		self.hi = 1


	def Kernel(self, ktype, dimx, dimy,  Ndim= 3,param1 = 1, param2 = 0 , param3 = 1, param4 = 1):
		#This will create a kernel depending on the type selected
		#param1 is sigma for gauss

		if Ndim%2 == 0:											#error check for even kernel size
			Ndim += 1
			print 'The Ndim provided is even, next highest odd was used' 

		if (ktype=='gauss') or (ktype =='Gauss') or (ktype =='gaussian') or (ktype =='Gaussian'):
			Sigma = param1
			kernel = np.zeros((Ndim, Ndim)) 					# create a kernel that is square with side dim 2*N+1
			mid = np.floor(len(kernel)/2)						#create a middle point index

			if dimx==1 and dimy==1:
				a =(1./(2.*pi*Sigma**2))
				for c in range(0, len(kernel[0])):
					for r in range(0, len(kernel)):
						kernel[r,c] = a*exp(-((r-mid)**2+(c-mid)**2)/(2.*Sigma**2))

			else:
				kernel = np.zeros(Ndim)
				mid = np.floor(Ndim/2)
				a = (1./(Sigma *(2*pi)**0.5))
				for p in range(Ndim):
					kernel[p] = a*exp((-0.5)*((p-mid)/Sigma)**2)
					print kernel[p]

				if dimy ==1:
					kernel.shape = ((Ndim,1))
					return kernel
				else:
					kernel.shape == ((1,Ndim))
					return kernel


		elif (ktype== 'gabor') or (ktype =='Gabor'):
			#WIP
			#param1 is sigma
			#param2 is gamma is the amount of stretch in the y direction ?
			#param3 is the Lambda representing wavelength
			#param4 is the gamma or the rotaation
			#param5 is psi which is the phase shift, we will ignore that here for now
			sigma = param1
			gamma = param2
			Lambda = param3
			theta = param4
			psi = 0
			mid = np.floor(Ndim/2)
			kernel = np.zeros((Ndim,Ndim))
			if dimx==1 and dimy==1:
				for c in range(0, len(kernel[0])):
					for r in range(0, len(kernel)):
						xp = (r- mid)*cos(theta) + (c-mid)*sin(theta)
						yp = -(r-mid)*sin(theta) + (c-mid)*cos(theta)
						b = exp(-(xp**2 + (gamma**2)*(yp**2))/(2*sigma**2))
						#kernel_imag[r,c] = gimag = b*sin(2*pi*xp/gamma + psi)
						kernel[r,c] = b*cos(2*pi*xp/gamma + psi)		

		elif (ktype == 'sobel') or (ktype =='Sobel'):
			#Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
			#Gx = np.rot90(Gy)or np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
			if dimx == 1 and dimy==1:
				kernel = np.array([[1,0,-1],[0,0,0],[-1,0,1]])
			elif dimy == 1:
				kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
			elif dimx == 1:
				kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

		else:
			print 'ktype parameter requires either "gauss", "gabor", or "sobel" '

		return kernel
	def something(self, img_src):
		self.raw  = np.copy(img_src)
		self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


