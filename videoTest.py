import cv2, dlib
import numpy as np
from imutils import face_utils
from tensorflow.keras.models import load_model
import winsound
import os

IMG_SIZE = (64,56)
B_SIZE = (64, 56)
margin = 95

left_eye_index = [17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41]
right_eye_index = [22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47]
root_dir = os.getcwd()
file_dir = root_dir + '\\face\\*'
process_path = root_dir + '\\process\\'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(root_dir + '\\shape_predictor_68_face_landmarks.dat')

font_letter = cv2.FONT_HERSHEY_PLAIN
model = load_model(root_dir + 'ls\\models\\rnn_model.h5')


def detect_gaze(eye_img):
    pred_l = model.predict(eye_img)
    accuracy = int(np.array(pred_l).max() * 100)
    print(accuracy)
    return accuracy
   
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

cap = cv2.VideoCapture(0) 
# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")

while(True):
  ret, frame = cap.read()
 
  if ret == True: 
     
    
    # Display the resulting frame    
    cv2.waitKey(10)
    cv2.imshow('frame',frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:  

      shapes = predictor(gray, face)
    
    for n in left_eye_index:
      x = shapes.part(n).x
      y = shapes.part(n).y

      next_point = n+1
      if n==41:
        next_point = 36 
            
      x2 = shapes.part(next_point).x
      y2 = shapes.part(next_point).y
      
      cv2.line(frame,(x,y),(x2,y2),(0,69,255),2)

    for n in right_eye_index:
      x = shapes.part(n).x
      y = shapes.part(n).y
      next_point = n+1
            
      if n==47:
        next_point = 42 
            
      x2 = shapes.part(next_point).x
      y2 = shapes.part(next_point).y
      cv2.line(frame,(x,y),(x2,y2),(153,0,153),2)
      
    shape_handle = face_utils.shape_to_np(shapes)
   
    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shape_handle[left_eye_index])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shape_handle[right_eye_index])
   
    eye_img_l_view = cv2.resize(eye_img_l, dsize=(128,112))
    eye_img_l_view = cv2.cvtColor(eye_img_l_view,cv2.COLOR_BGR2RGB)
    eye_img_r_view = cv2.resize(eye_img_r, dsize=(128,112))
    eye_img_r_view = cv2.cvtColor(eye_img_r_view, cv2.COLOR_BGR2RGB)
   
    eye_blink_left = cv2.resize(eye_img_l.copy(), B_SIZE)
    eye_blink_right = cv2.resize(eye_img_r.copy(), B_SIZE)
    eye_blink_left_i = eye_blink_left.reshape((1, B_SIZE[1], B_SIZE[0], 1)).astype(np.float32) / 255.
    eye_blink_right_i = eye_blink_right.reshape((1, B_SIZE[1], B_SIZE[0], 1)).astype(np.float32) / 255.
    
    left_acc = detect_gaze(eye_blink_left_i)
    right_acc = detect_gaze(eye_blink_right_i)
    output = np.zeros((900,820,3), dtype="uint8")
    output = cv2.line(output,(400,200), (400,0),(0,255,0),thickness=2)
    cv2.putText(output, "LEFT EYE: " + str(left_acc),(10,180), font_letter,1, (255,255,51),1)
    cv2.putText(output,"RIGHT EYE" + str(right_acc),(200,180), font_letter,1, (255,255,51),1)
    cv2.imshow("SAMPLE IMAGE", output)
    print("left accuracy => ", left_acc)
    print("right accuracy => ", right_acc)
    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else:
    break 
 
# When everything done, release the video capture and video write objects
cap.release()

# Closes all the frames
cv2.destroyAllWindows() 