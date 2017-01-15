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

def load_data():
    data = matrix(genfromtxt('spambase_data.csv', delimiter=','))
    data = np.array(data)

    return data


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


class Kernel:

	def __init__(self):
		self.Sigma = 1

	def gabor(self, Nsize, Sigma, Lambda, theta, gamma =1, psi =0):
		'''
		Gabor filter in 2D
		#sigma is the standard deviation
		#gamma is the  aspect ratio to make filter an ellipse
		#Lambda is the wavelength
		#theta or the rotaation
		#psi which is the phase shift, we will ignore for now
		'''
		
		if Nsize%2 == 0:											#error check for even kernel size
			Nsize += 1
			print 'The Ndim provided is even, next highest odd was used' 

		mid = np.floor(Nsize/2)
		kernel = np.zeros((Nsize,Nsize))
		for c in range(0, len(kernel[0])):
			for r in range(0, len(kernel)):
				xp = (r- mid)*cos(theta) + (c-mid)*sin(theta)
				yp = -(r-mid)*sin(theta) + (c-mid)*cos(theta)
				b = exp(-(xp**2 + (gamma**2)*(yp**2))/(2*Sigma**2))
				#kernel_imag[r,c] = gimag = b*sin(2*pi*xp/gamma + psi)
				kernel[r,c] = b*cos(2*pi*xp/Lambda + psi)

		return kernel/np.sum(kernel*kernel)


	def sobel(self, dimx, dimy):
		#Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
		#Gx = np.rot90(Gy)or np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
		if dimx == 1 and dimy==1:
			kernel = np.array([[1,0,-1],[0,0,0],[-1,0,1]])
		elif dimy == 1:
			kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
		elif dimx == 1:
			kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

		return kernel


	def gauss(self, Nsize, dimx, dimy, Sigma):
		if Nsize%2 == 0:											#error check for even kernel size
			Nsize += 1
			print 'The Ndim provided is even, next highest odd was used' 

		mid = np.floor(Nsize/2)

		if dimx==1 and dimy==1:
			kernel = np.zeros((Nsize, Nsize)) 					# create a kernel that is square with side dim 2*N+1
			a =(1./(2.*pi*Sigma**2))
			for c in range(0, len(kernel[0])):
				for r in range(0, len(kernel)):
					kernel[r,c] = a*exp(-((r-mid)**2+(c-mid)**2)/(2.*Sigma**2))
			return kernel

		else:
			kernel = np.zeros(Nsize)
			a = (1./(Sigma *(2*pi)**0.5))
			for p in range(Nsize):
				kernel[p] = a*exp((-0.5)*((p-mid)/Sigma)**2)

			if dimy ==1:
				kernel.shape = ((Nsize,1))
				return kernel
			else:
				kernel.shape == ((1,Nsize))
				return kernel



def convolve(mat, mask):
	#masks are either 2D or 1D
	#matrices can be 2D or multidimensional, need to loop through if 
	#check the mask and create a case scenario
	mat = np.array(mat)
	mask = np.array(mask)

	len_mat =  len(mat.shape)
	pad = np.floor(np.max(mask.shape)/2)
	mask_len = pad*2+1
	#this will get tricky if 1D masks


	if len(mat.shape) > 2:											#checking for high dimension
		depth, r_mat, c_mat = mat.shape
		out_mat = np.ones((depth,r_mat,c_mat))				#create a blank same dim as input
	else:
		r_mat, c_mat = mat.shape
		out_mat = np.ones((r_mat, c_mat))
		depth =1

	for dim in range(depth):
		ref_mat = 255*np.ones((r_mat+2*pad,c_mat+2*pad))		#create a ref mat with padding
		ref_mat[pad:r_mat+pad,pad:c_mat+pad] = np.copy(mat)		#fill center with raw mat
		print ref_mat
		temp_mat = np.ones((r_mat, c_mat))
		for r in range(r_mat):
			for c in range(c_mat):
				temp_mat[r,c] = np.sum(np.multiply(mask, ref_mat[r:r+mask_len,c:c+mask_len]))
		if depth>1:		
			out_mat[dim] = np.copy(temp_mat)
		else:
			out_mat = np.copy(temp_mat)
	return out_mat




def something(img_src):
	raw  = np.copy(img_src)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


