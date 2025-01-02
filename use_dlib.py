import os
import cv2
import dlib
import numpy as np

# Load the pre-trained facial landmark detector from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmark_train(direction):
    
    images=[]
    for i in os.listdir(direction):
        images.append(i)
        
       
    normal_eyelash_distance = np.zeros(len(images))
    normal_eyebrow_height = np.zeros(len(images))
    normal_mouth_height = np.zeros(len(images))
    normal_mouth_width = np.zeros(len(images))

    for i in range(len(images)):
        image_path = os.path.join(direction, images[i])
        image = cv2.imread(image_path)
        
        
        # Detect faces in the image
        faces = detector(image)
        
        for face in faces:
            # Detect facial landmarks
            landmarks = predictor(image, face)
            
            # Calculate the distance between the eyelash landmarks
            up_eyelash = landmarks.part(45-1)  
            down_eyelash = landmarks.part(47-1)  
            eyelash_distance = np.sqrt((down_eyelash.x - up_eyelash.x) ** 2 + (down_eyelash.y - up_eyelash.y) ** 2)
            # Normalize the distance for every image
            up_face1 = landmarks.part(25-1) 
            down_face1 = landmarks.part(11-1) 
            distance1 = np.sqrt((down_face1.x - up_face1.x) ** 2 + (down_face1.y - up_face1.y) ** 2)
            normal_eyelash_distance[i] = 1000*(eyelash_distance / distance1)


            # Calculate the distance between the eyebrow landmarks
            eyebrow1 = landmarks.part(25-1)  
            eyelash1 = landmarks.part(45-1)  
            eyebrow_height = np.sqrt((eyelash1.x - eyebrow1.x) ** 2 + (eyelash1.y - eyebrow1.y) ** 2)
            # Normalize the distance for every image
            normal_eyebrow_height[i] = 1000*(eyebrow_height / distance1)
            
            
            # Calculate the distance between the mouth height landmarks
            up_lip = landmarks.part(52-1)  
            down_lip = landmarks.part(58-1)  
            mouth_height1 = np.sqrt((down_lip.x - up_lip.x) ** 2 + (down_lip.y - up_lip.y) ** 2)
            # Normalize the distance for every image
            up_face2 = landmarks.part(28-1) 
            down_face2 = landmarks.part(9-1) 
            distance2 = np.sqrt((down_face2.x - up_face2.x) ** 2 + (down_face2.y - up_face2.y) ** 2)
            normal_mouth_height[i] = 1000*(mouth_height1 / distance2)
            
            
            # Calculate the distance between the mouth width landmarks
            right_lip = landmarks.part(55-1)  
            left_lip = landmarks.part(49-1)  
            mouth_width1 = np.sqrt((left_lip.x - right_lip.x) ** 2 + (left_lip.y - right_lip.y) ** 2)
            # Normalize the distance for every image
            right_face = landmarks.part(13-1) 
            left_face = landmarks.part(5-1) 
            distance3 = np.sqrt((right_face.x - left_face.x) ** 2 + (right_face.y - left_face.y) ** 2)
            normal_mouth_width[i] = 1000*(mouth_width1 / distance3)
    
    return normal_eyelash_distance, normal_eyebrow_height, normal_mouth_height, normal_mouth_width

def get_landmark_test(image):
    
    image = cv2.imread(image)
    # image= cv2.resize(image, (image.shape[1]*5,image.shape[0]*5))
    
    # Detect faces in the image
    faces = detector(image)
    
    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(image, face)
        
        # Calculate the distance between the eyelash landmarks
        up_eyelash = landmarks.part(45-1)  
        down_eyelash = landmarks.part(47-1)  
        eyelash_distance = np.sqrt((down_eyelash.x - up_eyelash.x) ** 2 + (down_eyelash.y - up_eyelash.y) ** 2)
        # Normalize the distance for every image
        up_face1 = landmarks.part(25-1) 
        down_face1 = landmarks.part(11-1) 
        distance1 = np.sqrt((down_face1.x - up_face1.x) ** 2 + (down_face1.y - up_face1.y) ** 2)
        normal_eyelash_distance = 1000*(eyelash_distance / distance1)


        # Calculate the distance between the eyebrow landmarks
        eyebrow1 = landmarks.part(25-1)  
        eyelash1 = landmarks.part(45-1)  
        eyebrow_height = np.sqrt((eyelash1.x - eyebrow1.x) ** 2 + (eyelash1.y - eyebrow1.y) ** 2)
        # Normalize the distance for every image
        normal_eyebrow_height = 1000*(eyebrow_height / distance1)
        
        
        # Calculate the distance between the mouth height landmarks
        up_lip = landmarks.part(52-1)  
        down_lip = landmarks.part(58-1)  
        mouth_height1 = np.sqrt((down_lip.x - up_lip.x) ** 2 + (down_lip.y - up_lip.y) ** 2)
        # Normalize the distance for every image
        up_face2 = landmarks.part(28-1) 
        down_face2 = landmarks.part(9-1) 
        distance2 = np.sqrt((down_face2.x - up_face2.x) ** 2 + (down_face2.y - up_face2.y) ** 2)
        normal_mouth_height = 1000*(mouth_height1 / distance2)
        
        
        # Calculate the distance between the mouth width landmarks
        right_lip = landmarks.part(55-1)  
        left_lip = landmarks.part(49-1)  
        mouth_width1 = np.sqrt((left_lip.x - right_lip.x) ** 2 + (left_lip.y - right_lip.y) ** 2)
        # Normalize the distance for every image
        right_face = landmarks.part(13-1) 
        left_face = landmarks.part(5-1) 
        distance3 = np.sqrt((right_face.x - left_face.x) ** 2 + (right_face.y - left_face.y) ** 2)
        normal_mouth_width = 1000*(mouth_width1 / distance3)

    return normal_eyelash_distance, normal_eyebrow_height, normal_mouth_height, normal_mouth_width

  