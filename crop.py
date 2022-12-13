  # image extract python
import cv2, dlib, glob
import numpy as np
from imutils import face_utils
from PIL import Image
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing import image
import os

left_eye_index = [17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41]
right_eye_index = [22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47]
root_dir = os.getcwd()
file_dir = root_dir + '\\face\\*'
process_path = root_dir + '\\process\\'

IMG_SIZE = (56, 64)
B_SIZE = (34, 26)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(root_dir + '\\shape_predictor_68_face_landmarks.dat')

def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect  

for filename in glob.glob(file_dir):
    print("??")
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:  

      shapes = predictor(gray, face)
      
    # for n in range(36,42):
    for n in left_eye_index:
      x = shapes.part(n).x
      y = shapes.part(n).y

      next_point = n+1
      if n==41:
        next_point = 36 
            
      x2 = shapes.part(next_point).x
      y2 = shapes.part(next_point).y
      
      cv2.line(image,(x,y),(x2,y2),(0,69,255),2)

    for n in right_eye_index:
      x = shapes.part(n).x
      y = shapes.part(n).y
      next_point = n+1
            
      if n==47:
        next_point = 42 
            
      x2 = shapes.part(next_point).x
      y2 = shapes.part(next_point).y
      cv2.line(image,(x,y),(x2,y2),(153,0,153),2)
      
    shape_handle = face_utils.shape_to_np(shapes)
   
    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shape_handle[left_eye_index])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shape_handle[right_eye_index])
   
    eye_img_l_view = cv2.resize(eye_img_l, dsize=(128,112))
    eye_img_l_view = cv2.cvtColor(eye_img_l_view,cv2.COLOR_BGR2RGB)
    eye_img_r_view = cv2.resize(eye_img_r, dsize=(128,112))
    eye_img_r_view = cv2.cvtColor(eye_img_r_view, cv2.COLOR_BGR2RGB)
   
    eye_blink_left = cv2.resize(eye_img_l.copy(), B_SIZE)
    eye_blink_right = cv2.resize(eye_img_r.copy(), B_SIZE)
   
    # plt.imshow(eye_img_l_view, interpolation='nearest')
    # plt.show()
    ri = random.randint(0, 100000)
    
    plt.imsave(process_path + str(ri)+"l.jpg", eye_img_l_view)
    plt.imsave(process_path + str(ri) + "r.jpg", eye_img_r_view)
