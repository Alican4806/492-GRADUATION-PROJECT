
import ultralytics
from ultralytics import YOLO
import numpy as np
import torch

from matplotlib import pyplot as plt

model = YOLO("best.pt")


  
import os
import cv2 

import time




# init camera
execution_path = os.getcwd()
camera = cv2.VideoCapture(0)
pTime = 0 # It is previous time and it was 0 at started
cTime = 0 
while True:
    # Init and FPS process
    start_time = time.time()

    # Grab a single frame of video
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    cTime = time.time() # we got current time as cTime
    
    fps = 1/ (cTime-pTime) # We found fps
    
    pTime = cTime # cTime assigned as previous time 
    
    cv2.putText(frame,str(int(fps)),(10,100),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3) # we write fps value to the screen.
    
    results = model.predict(frame,conf = 0.7) # We are adjusting confidence value

    
    for r in results:
        
        boxes = r.boxes
        
        for box in boxes:
            
            
            class_id = "Fire"
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(),2)
            print("Object type:", class_id)
            print("Coordinates:", cords)
            print("Probability:", conf)
            print("---")
            if conf>0.5: # If confidence value is greater than 0.5, the drawing process is true.
                cv2.rectangle(frame,(cords[0],cords[1]),(cords[2],cords[3]),(0,0,255),2) # The coordinates are drawn as rectangulars
                cv2.putText(frame, f"{class_id} {conf}", (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # The confidence level and its explanation are written
        cv2.imshow("Shown",frame)
        


            
            #conf = box.conf
            # b = box.xyxy[0]  # box koordinatlarını (sol, üst, sağ, alt) formatında alın
            # c = box.cls
            #print(box.conf)
            # Algılama güvenilirliğini yazdırın
            #cv2.putText(frame, f'Confidence: {conf}', (b, c - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
           # annotator.box_label(b, model.names[int(c)])  # bounding box ve etiketi çizdirin
        #img = annotator.result()

   


    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

camera.release()
cv2.destroyAllWindows()
"""





   '''''' 
    print(len(results[0]))
    
    if len(results[0]) > 0:
        print(results[0])

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(16) & 0xFF == ord('q'):
        break

# Release handle to the webcam
camera.release()
cv2.destroyAllWindows()'''
"""