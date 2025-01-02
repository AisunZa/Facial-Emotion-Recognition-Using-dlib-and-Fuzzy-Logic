import cv2
import use_dlib
import numpy as np 
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Getting training parameters
direction1 = r'C:/Users/Surface/Desktop/lessons/soft computing/p1/train/angry'
normal_angry_eyelash_distance, normal_angry_eyebrow_height, normal_angry_mouth_height, normal_angry_mouth_width = use_dlib.get_landmark_train(direction1)
direction2 = r'C:/Users/Surface/Desktop/lessons/soft computing/p1/train/happy'
normal_happy_eyelash_distance, normal_happy_eyebrow_height, normal_happy_mouth_height, normal_happy_mouth_width = use_dlib.get_landmark_train(direction2)
direction3 = r'C:/Users/Surface/Desktop/lessons/soft computing/p1/train/sad'
normal_sad_eyelash_distance, normal_sad_eyebrow_height, normal_sad_mouth_height, normal_sad_mouth_width = use_dlib.get_landmark_train(direction3)
direction4 = r'C:/Users/Surface/Desktop/lessons/soft computing/p1/train/disgusted'
normal_disgusted_eyelash_distance, normal_disgusted_eyebrow_height, normal_disgusted_mouth_height, normal_disgusted_mouth_width = use_dlib.get_landmark_train(direction4)
direction5 = r'C:/Users/Surface/Desktop/lessons/soft computing/p1/train/surprised'
normal_surprised_eyelash_distance, normal_surprised_eyebrow_height, normal_surprised_mouth_height, normal_surprised_mouth_width = use_dlib.get_landmark_train(direction5)
direction6 = r'C:/Users/Surface/Desktop/lessons/soft computing/p1/train/frightened'
normal_frightened_eyelash_distance, normal_frightened_eyebrow_height, normal_frightened_mouth_height, normal_frightened_mouth_width = use_dlib.get_landmark_train(direction6)


mouth_widths = np.concatenate((normal_frightened_mouth_width, normal_sad_mouth_width, normal_angry_mouth_width,  
                               normal_surprised_mouth_width, normal_happy_mouth_width, normal_disgusted_mouth_width), axis=0)
mouth_heights = np.concatenate((normal_frightened_mouth_height, normal_sad_mouth_height, normal_angry_mouth_height,  
                               normal_surprised_mouth_height, normal_happy_mouth_height, normal_disgusted_mouth_height), axis=0)
eyebrow_heights = np.concatenate((normal_frightened_eyebrow_height, normal_sad_eyebrow_height, normal_angry_eyebrow_height,  
                               normal_surprised_eyebrow_height, normal_happy_eyebrow_height, normal_disgusted_eyebrow_height), axis=0)
eyelash_distances = np.concatenate((normal_frightened_eyelash_distance, normal_sad_eyelash_distance, normal_angry_eyelash_distance,  
                               normal_surprised_eyelash_distance, normal_happy_eyelash_distance, normal_disgusted_eyelash_distance), axis=0)




# Generate universe variables
mouth_width = ctrl.Antecedent(np.arange(0, max(mouth_widths) + 1, 1), 'mouth_width')
mouth_height = ctrl.Antecedent(np.arange(0, max(mouth_heights) + 1, 1),'mouth_height')
eyelash = ctrl.Antecedent(np.arange(0, max(eyelash_distances) + 1, 1),'eyelash')
eyebrow = ctrl.Antecedent(np.arange(0, max(eyebrow_heights) + 1, 1),'eyebrow')
emotion = ctrl.Consequent(np.arange(-0.1, 1.2, 0.1),'emotion')


# Generate fuzzy membership functions
mouth_width.automf(3)
mouth_height.automf(3)
eyelash.automf(3)
eyebrow.automf(3)


emotion['happy'] = fuzz.trapmf(emotion.universe, [-0.1, -0.1, 0, 0.2])
emotion['surprised'] = fuzz.trimf(emotion.universe, [0, 0.2, 0.4])
emotion['frightened'] = fuzz.trimf(emotion.universe, [0.2, 0.4, 0.6])
emotion['angry'] = fuzz.trimf(emotion.universe, [0.4, 0.6, 0.8])
emotion['disgusted'] = fuzz.trimf(emotion.universe, [0.6, 0.8, 1.0])
emotion['sad'] = fuzz.trapmf(emotion.universe, [0.8, 1.0, 1.1, 1.1])

rule1=ctrl.Rule(mouth_width['good'] & mouth_height['average'] & eyelash['poor'] & eyebrow['good'], emotion['happy'])
rule2=ctrl.Rule(mouth_width['poor'] & mouth_height['good'] & eyelash['good'] & eyebrow['good'], emotion['surprised'])
rule3=ctrl.Rule(mouth_width['average'] & mouth_height['good'] & eyelash['good'] & eyebrow['good'], emotion['frightened'])
rule4=ctrl.Rule(mouth_width['average'] & mouth_height['good'] & eyelash['poor'] & eyebrow['poor'], emotion['angry'])
rule5=ctrl.Rule(mouth_width['average'] & mouth_height['average'] & eyelash['poor'] & eyebrow['average'], emotion['disgusted'])
rule6=ctrl.Rule(mouth_width['good'] & mouth_height['poor'] & eyelash['average'] & eyebrow['poor'], emotion['sad'])


emotion_ctrl=ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
show_emotion=ctrl.ControlSystemSimulation(emotion_ctrl)


# Load Test image
image_path = 'angry.jpg'
img = cv2.imread(image_path)
# img = cv2.resize(img, (img.shape[1]*3,img.shape[0]*3))
cv2.imshow('image',img)
test_eyelash_distance, test_eyebrow_height, test_mouth_height, test_mouth_width = use_dlib.get_landmark_test(image_path)

show_emotion.input['mouth_width']=test_mouth_width
show_emotion.input['mouth_height']=test_mouth_height
show_emotion.input['eyelash']=test_eyelash_distance
show_emotion.input['eyebrow']=test_eyebrow_height


show_emotion.compute()

print(show_emotion.output['emotion'])

emotion.view(sim=show_emotion)

