import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
def warp_frame(frame1, frame2, p):
  affine_transformation_matrix = np.array([ [p[0], p[1], p[2]],
                                            [p[3], p[4], p[5]],
                                            [0, 0, 1] ]) 
  warped_image = np.zeros((frame1.shape[0], frame1.shape[1]))
  for r in range(0,frame1.shape[0]):
    for c in range(0,frame1.shape[1]):
      warped_image_coord = homog_to_hetrog(np.dot(affine_transformation_matrix, [r, c, 1]))
      # print(warped_image_coord)
      warped_image_coord = warped_image_coord.astype(int)
      if(warped_image_coord[0]>0 and warped_image_coord[0]< warped_image.shape[0] and warped_image_coord[1]>0 and warped_image_coord[1]<warped_image.shape[1]):
        warped_image[r][c] = frame2[warped_image_coord[0]][warped_image_coord[1]]
  return warped_image

def warp_point(x, y, p):
  affine_transformation_matrix = np.array([ [p[0], p[1], p[2]],
                                            [p[3], p[4], p[5]],
                                            [0, 0, 1] ]) 

  warped_coord = homog_to_hetrog(np.dot(affine_transformation_matrix, [y, x, 1]))
  warped_coord = warped_coord.astype(int)
  return warped_coord[0], warped_coord[1]

def compute_error_image(frame, warped_frame, x, y, w, h):
  template = extract_patch(frame, x, y, w, h)
  warped_patch = extract_patch(warped_frame, x, y, w, h)
  return template - warped_patch

#takes in warped image
def find_gradients(frame):
  sobel_x = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
  sobel_y = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)
  return sobel_x, sobel_y

def homog_to_hetrog(point):
    point = point / point[-1]
    return point[:-1]

def additive_alignment(frame1, frame2, x1,y1, x2, y2, p):
  iterations = 0
  delta_p = 1000
  while(np.linalg.norm(delta_p) > 1 ):
    warped_frame = warp_frame(frame1, frame2, p)
    # print("warped_frame: ",frame1)
    grad_x, grad_y = find_gradients(warped_frame)
    hessian = np.zeros((6,6))
    steepest_descent_with_error = np.zeros((6,1))
    # print("grad_x: ",grad_x)
    for r in range(y1, y2):
      for c in range(x1,x2):
        steepest_descent = np.dot(np.array([grad_x[r,c], grad_y[r,c]]),np.array([ [c,r,1,0,0,0],
                                                                                [0,0,0,c,r,1] ])).reshape((1,6))
        # steepest_descent = np.dot(np.array([grad_x[r,c], grad_y[r,c]]),np.array([ [1,0,c,r,0,0],
        #                                                                         [0,1,0,0,c,r] ])).reshape((1,6))
        # steepest_descent = np.dot(np.array([grad_x[r,c], grad_y[r,c]]),np.array([ [c,0,r,0,1,0],
        #                                                                         [0,c,0,r,0,1] ])).reshape((1,6))
        hessian =+ np.dot(steepest_descent.T,steepest_descent)
        #bos 3al r,c
        steepest_descent_with_error += steepest_descent.T * (warped_frame[r,c] -frame1[r,c] )
    delta_p = np.dot(np.linalg.pinv(hessian),steepest_descent_with_error).reshape(6)
    # print("hessian: ", hessian)
    # print("delta_P: ",delta_p)
    p += delta_p
    print("p: ",p)
    iterations += 1

  return p

def lucas_kanade_tracker(frames, x1,y1, x2, y2, p):
  window_params = []
  window_params.append(np.array([x1,y1,x2,y2]))
  for t in range(frames.shape[0]-1):
    print("Frame: ", t)
    print("HERE IS X1:", x1)
    # p = [ 0.97042495,  0.0268497,   0.83223983,  0.01766428,  1.0044413,  -4.30594529]
    p = additive_alignment(frames[t], frames[t+1], x1,y1, x2, y2, p)
    x1,y1 = warp_point(x1,y1,p)
    # print("x1 after warp_pointt:",x1)
    x2,y2 = warp_point(x2,y2,p)
    window_params.append(np.array([x1,y1,x2,y2]))
  return np.array(window_params)