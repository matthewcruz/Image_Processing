#!/usr/bin/env python

#this will practice eigenfaces using a set of 10

import numpy as np
import os
import cv2 as cv2
import cv2.cv as cv

#step one, walk through file folder called '/home/mtcruz/Desktop/OpenCV_work/ten_face'
#determine number of images and names of files
#loop through each and convert to greyscale then vectorize those values into a column
#add these columns to the existing eigenface Matrix


def load_img_mat(walk_dir):
  print "Running image load"
  #walk_dir  = '/home/mtcruz/Desktop/OpenCV_work/ten_faces'
  img_mat = np.empty([0,0],dtype = np.uint8)
  #use os to walk folder and find all images inside, then loop through files
  for root_dir, subdirs, files in os.walk(walk_dir, topdown=True):
    numOfFiles = len(files)
    print subdirs
    for filename in files: #***loop through each file
      img_path = os.path.join(walk_dir,filename)
      img = cv2.imread(img_path, 0)
      img = cv2.resize(img, None, fx =0.2, fy = 0.2, interpolation = cv2.INTER_AREA )
      row_len = len(img)
      img_vec = np.empty([0,0], dtype = np.uint8) #initialize an empty image vector
      #***loop through each column converting to vectors in the image and stack vertically
      for col in img.T:
        col = np.reshape(np.array(col), (row_len,1))
        if len(img_vec) >0:         #*** special handler for first time through loop
          img_vec = np.vstack((img_vec,col))
          #img_vec = img_vec.reshape((row_len,1))
        else:
          img_vec = col*1
      #***build image_matrix from img_vectors
      if len(img_mat) >0:
        img_mat = np.hstack((img_mat,img_vec))
        #print img_mat.shape
      else:
        img_mat = img_vec*1
      num_imgs=img_mat.shape[1]

  return img_mat.T, row_len, img_count

  # print img_mat, numOfFiles


def compute_eig(row_len, img_mat):
  mean_array = np.array(np.mean(img_mat, axis=1), dtype = int)
  mean_array = mean_array.reshape((row_len**2,1))
  imgarray = np.subtract(img_mat, mean_array)
  img_cov = np.cov(imgarray.T)
  eigval, eigvec = np.linalg.eig(img_cov) 

  #sort eigenvalues in descending order
  eigval_sorted = np.sort(eigval)[::-1]
  eigvec_sorted = eigvec[:,eigval.argsort()[::-1]]
  #eigfaces = np.dot(eigvec_sorted,imgarray.T)
  eigfaces = np.dot(imgarray, eigvec_sorted.T)

  eigfaces = eigfaces.T
  counter = 0
  eigfaces = eigfaces + mean_array.T


  build = np.empty((0,0))
  #eigfaces_norm = np.add(eigfaces[0], mean_array).T
  while counter <250:
    build_temp = np.reshape(mean_array.T[0][counter*row_len: (counter+1)*row_len], (row_len,1))
    if len(build) == 0:
      build = build_temp
    else:
      build = np.hstack((build,build_temp))
    counter +=1

  build = np.array(build, dtype = np.uint8)
  cv2.imshow('ghostly', build)

  return eigvec_sorted, mean_array, eigfaces

def covert_input_img(img_mat, eigvec_sorted, mean_array, eigfaces):
  #load the image and convert using the method above
  walk_dir  = '/home/mtcruz/Desktop/OpenCV_work/'
  filename = 'Tom_Daschle_0012.jpg'
  img_dir = os.path.join(walk_dir,filename)
  test_img = cv2.imread(img_dir, 0)
  #convert the test image and subtract the mean array from it
  #note row_len must be the same length as training images
  img_vec = np.empty([0,0], dtype = np.uint8) #initialize an empty image vector
  row_len = len(test_img)

  for col in test_img.T:
    col = np.reshape(np.array(col), (row_len,1))
    if len(img_vec) >0:         #*** special handler for first time through loop
      img_vec = np.vstack((img_vec,col))
      #img_vec = img_vec.reshape((row_len,1))
    else:
      img_vec = col*1

  #calculate tesst vec by by subtracting mean
  test_vec = np.subtract(img_vec, mean_array)

  #may use the eigenvec matrix (10x10)
  #this step should convert face data into eigenface data by projecting into ..
  #eigenface space
  #each face should then have the result of each vector
  #This vector.this image
  #10x1.1x250 + 
  #10x10x10X250

  #find wk = uk.T(img_vec-mean_array).250x10

def face_color():
  walk_dir  = '/home/mtcruz/Desktop/OpenCV_work/'
  filename = 'Tom_Daschle_0001.jpg'
  img_path = os.path.join(walk_dir,filename)
  img_mat = np.empty([0,0],dtype = np.uint8)
  img = cv2.imread(img_path)
  

def full_conversion():
  from sklearn.grid_search import GridSearchCV
  from sklearn.decomposition import RandomizedPCA
  from sklearn.svm import SVC

  ###load the training images
  walk_dir  = '/home/mtcruz/Desktop/OpenCV_work/ten_faces'
  img_mat, row_len, img_count = load_img_mat(walk_dir)

  #make sure they are in the correct format and size
  #done
  #make  sure they have the correct name labels, and create an array that has the correct integer labels
  #compute a pca on the converted data 
  #convert the data using the eigenvectors 
  #using the grid search function create a parameter grid over which to search and find best values
  #using the eigenfaces compute the SVM using the rbf kernel and the found parameter
  #use the resulting thing to predict an unknown image that must be correctly formatted.
