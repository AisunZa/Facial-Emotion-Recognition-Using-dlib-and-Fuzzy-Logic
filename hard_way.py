import cv2
import use_dlib
import numpy as np 
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


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
mouth_width = np.arange(min(mouth_widths), max(mouth_widths) + 1, 1)
mouth_height = np.arange(min(mouth_heights), max(mouth_heights) + 1, 1)
eyebrow = np.arange(min(eyebrow_heights), max(eyebrow_heights) + 1, 1)
eyelash = np.arange(min(eyelash_distances), max(eyelash_distances) + 1, 1)
emotion = np.arange(-0.1, 1.2, 0.1)


# Generate fuzzy membership functions
mouth_width_small = fuzz.trimf(mouth_width, [min(mouth_widths), min(mouth_widths), np.mean(mouth_widths)])
mouth_width_normal = fuzz.trimf(mouth_width, [min(mouth_widths), np.mean(mouth_widths), max(mouth_widths)])
mouth_width_big = fuzz.trimf(mouth_width, [np.mean(mouth_widths), max(mouth_widths), max(mouth_widths)])

mouth_height_small = fuzz.trimf(mouth_height, [min(mouth_heights), min(mouth_heights), np.mean(mouth_heights)])
mouth_height_normal = fuzz.trimf(mouth_height, [min(mouth_heights), np.mean(mouth_heights), max(mouth_heights)])
mouth_height_big = fuzz.trimf(mouth_height, [np.mean(mouth_heights), max(mouth_heights), max(mouth_heights)])

eyelash_small = fuzz.trimf(eyelash, [min(eyelash_distances), min(eyelash_distances), np.mean(eyelash_distances)])
eyelash_normal = fuzz.trimf(eyelash, [min(eyelash_distances), np.mean(eyelash_distances), max(eyelash_distances)])
eyelash_big = fuzz.trimf(eyelash, [np.mean(eyelash_distances), max(eyelash_distances), max(eyelash_distances)])

eyebrow_small = fuzz.trimf(eyebrow, [min(eyebrow_heights), min(eyebrow_heights), np.mean(eyebrow_heights)])
eyebrow_normal = fuzz.trimf(eyebrow, [min(eyebrow_heights), np.mean(eyebrow_heights), max(eyebrow_heights)])
eyebrow_big = fuzz.trimf(eyebrow, [np.mean(eyebrow_heights), max(eyebrow_heights), max(eyebrow_heights)])


happy = fuzz.trapmf(emotion, [-0.1, -0.1, 0, 0.2])
surprised = fuzz.trimf(emotion, [0, 0.2, 0.4])
frightened = fuzz.trimf(emotion, [0.2, 0.4, 0.6])
angry = fuzz.trimf(emotion, [0.4, 0.6, 0.8])
disgusted = fuzz.trimf(emotion, [0.6, 0.8, 1.0])
sad = fuzz.trapmf(emotion, [0.8, 1.0, 1.1,1.1])



# Load test image
image_path = 'sad.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img, (img.shape[1]*3,img.shape[0]*3))
cv2.imshow('image',img)

# Get the test values from test image
test_eyelash_distance, test_eyebrow_height, test_mouth_height, test_mouth_width = use_dlib.get_landmark_test(image_path)


# Activate fuzzy membership functions
mouth_width_small1 = fuzz.interp_membership(mouth_width, mouth_width_small, test_mouth_width)
mouth_width_normal1 = fuzz.interp_membership(mouth_width, mouth_width_normal, test_mouth_width)
mouth_width_big1 = fuzz.interp_membership(mouth_width, mouth_width_big, test_mouth_width)

mouth_height_small1 = fuzz.interp_membership(mouth_height, mouth_height_small, test_mouth_height)
mouth_height_normal1 = fuzz.interp_membership(mouth_height, mouth_height_normal, test_mouth_height)
mouth_height_big1 = fuzz.interp_membership(mouth_height, mouth_height_big, test_mouth_height)

eyebrow_small1 = fuzz.interp_membership(eyebrow, eyebrow_small, test_eyebrow_height)
eyebrow_normal1 = fuzz.interp_membership(eyebrow, eyebrow_normal, test_eyebrow_height)
eyebrow_big1 = fuzz.interp_membership(eyebrow, eyebrow_big, test_eyebrow_height)

eyelash_small1 = fuzz.interp_membership(eyelash, eyelash_small, test_eyelash_distance)
eyelash_normal1 = fuzz.interp_membership(eyelash, eyelash_normal, test_eyelash_distance)
eyelash_big1 = fuzz.interp_membership(eyelash, eyelash_big, test_eyelash_distance)


# Apply the rules
rule_happy = np.fmin(mouth_width_big1,
              np.fmin(mouth_height_normal1,
              np.fmin(eyebrow_normal1, eyelash_normal1)))
activate_rule_happy = np.fmin(rule_happy, happy)

rule_surprised = np.fmin(mouth_width_small1,
                  np.fmin(mouth_height_normal1,
                  np.fmin(eyebrow_big1, eyelash_big1)))
activate_rule_surprised = np.fmin(rule_surprised, surprised)

rule_frightened = np.fmin(mouth_width_normal1,
                  np.fmin(mouth_height_big1,
                  np.fmin(eyebrow_normal1, eyelash_big1)))
activate_rule_frightened = np.fmin(rule_frightened, frightened)

rule_angry = np.fmin(mouth_width_normal1,
              np.fmin(mouth_height_big1,
              np.fmin(eyebrow_normal1, eyelash_normal1)))
activate_rule_angry = np.fmin(rule_angry, angry)

rule_disgusted = np.fmin(mouth_width_small1,
                  np.fmin(mouth_height_small1,
                  np.fmin(eyebrow_normal1, eyelash_small1)))
activate_rule_disgusted = np.fmin(rule_disgusted, disgusted)

rule_sad = np.fmin(mouth_width_small1,
            np.fmin(mouth_height_small1,
            np.fmin(eyebrow_normal1, eyelash_normal1)))
activate_rule_sad = np.fmin(rule_sad, sad)


# Aggregate all six output membership functions together
aggregated = np.fmax(activate_rule_happy,
              np.fmax(activate_rule_surprised,
              np.fmax(activate_rule_frightened,
              np.fmax(activate_rule_angry,
              np.fmax(activate_rule_disgusted,activate_rule_sad)))))


# Calculate defuzzified result
emotions = fuzz.defuzz(emotion, aggregated, 'centroid')
emotions_activation = fuzz.interp_membership(emotion, aggregated, emotions)  # for plot



# Visualize 
emotion0 = np.zeros_like(emotion)

fig, ax0 = plt.subplots(figsize=(12, 12))

ax0.fill_between(emotion, emotion0, activate_rule_happy, facecolor='b', alpha=0.7)
ax0.fill_between(emotion, emotion0, activate_rule_surprised, facecolor='g', alpha=0.7)
ax0.fill_between(emotion, emotion0, activate_rule_frightened, facecolor='m', alpha=0.7)
ax0.fill_between(emotion, emotion0, activate_rule_angry, facecolor='r', alpha=0.7)
ax0.fill_between(emotion, emotion0, activate_rule_disgusted, facecolor='y', alpha=0.7)
ax0.fill_between(emotion, emotion0, activate_rule_sad, facecolor='c', alpha=0.7)

ax0.plot(emotion, happy, 'b', linewidth=1.5 )
ax0.plot(emotion, surprised, 'g', linewidth=1.5)
ax0.plot(emotion, frightened, 'm', linewidth=1.5)
ax0.plot(emotion, angry, 'r', linewidth=1.5,)
ax0.plot(emotion, disgusted, 'y', linewidth=1.5)
ax0.plot(emotion, sad, 'c', linewidth=1.5)


ax0.plot([emotions, emotions], [0, emotions_activation], 'k', linewidth=1.5, alpha=0.9)

ax0.set_title('Aggregated membership and result (line)')
ax0.legend(['happy', 'surprised','frightened','angry','disgusted','sad'], fontsize=14)


# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()


