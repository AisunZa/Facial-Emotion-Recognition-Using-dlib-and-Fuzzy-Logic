"# Emotion-detection-using-fuzzy-system" 
"
Project Report: Facial Emotion Recognition Using dlib and Fuzzy Logic
Introduction
This project aims to develop a facial emotion recognition system that quantifies emotional expressions based on facial landmarks detected in images. The recognition process involves calculating specific facial measurements—such as mouth width, mouth height, eyelash distance, and eyebrow height—using a pre-trained facial landmark detection model from dlib. Subsequently, these measurements are evaluated using fuzzy logic to classify the emotion displayed in the image.

Objective
The primary objective of this project is to accurately identify and classify six different emotional states:

Angry
Happy
Sad
Disgusted
Surprised
Frightened
Methodology
1. Facial Landmark Detection
Facial landmarks are key points on a face that provide necessary measurements for emotion detection. For this project, the following landmarks were utilized:

Mouth Width and Height: Measured between specific lip landmarks.
Eyelash Distance: The distance between respective upper and lower eyelash landmarks.
Eyebrow Height: The vertical distance between eyebrow and eyelash landmarks.
The process involves the following steps:

Loading and Preprocessing Data:

The facial landmark detector is initialized using dlib’s pre-trained models.
Images are read and converted into a format suitable for analysis.
Landmark Calculation:

For each image in the training dataset, facial landmarks are located.
Distances for the specified features (eyelash distance, eyebrow height, mouth height, and mouth width) are calculated and normalized based on the face dimensions.
2. Training Dataset Creation
The project utilizes images of various emotions, which are stored in different subdirectories. Using the get_landmark_train function, the following datasets were created:

Angry
Happy
Sad
Disgusted
Surprised
Frightened
Each dataset stores normalized measurements for the respective emotion, later used for fuzzy logic inputs.

3. Fuzzy Logic Control System
The project employs a fuzzy logic system to classify emotions based on calculated measurements. The steps include:

Defining Fuzzy Membership Functions:

The fuzzy logic system consists of input variables (mouth width, mouth height, eyelash distance, eyebrow height) and an output variable (emotion classification).
The classification emotions were represented using trapezoidal and triangular membership functions.
Creating Fuzzy Rules:

Rules were formulated to determine the relationship between input variables and the emotional state. Examples include:
If mouth width is ‘good’ and mouth height is ‘average’ while eyelash distance is ‘poor’ and eyebrow height is ‘good’, then the emotion is classified as ‘happy’.
Simulation and Testing:

A ControlSystem and ControlSystemSimulation were created to implement and simulate the fuzzy logic rules.
Test images are processed using the get_landmark_test function, which implements the same feature extraction as in training.
4. Results and Outputs
A test image of a person exhibiting an "angry" emotion was processed to compute the respective distances for eyelash, eyebrow, mouth width, and height. After feeding these inputs into the fuzzy logic controller, the output emotion was derived and displayed.

5. Sample Output Interpretation
The system successfully classified the emotional state based on the given test image inputs. The computed results were visually represented using the view method, detailing the fuzzy logic representation of the emotion output.

6. Evaluation
The system's effectiveness can be evaluated by testing with real-world images and comparing predicted results against actual emotions expressed by individuals. Further enhancements could involve using a larger and more diverse dataset for improved accuracy and robustness.

Conclusion
The project successfully demonstrates a facial emotion recognition system leveraging dlib for facial landmark detection and fuzzy logic for emotional classification. This integration provides a flexible approach to emotion detection that can potentially be used in various applications, such as interactive AI systems, sentiment analysis, and emotional assistance technologies. Future work may include refining the fuzzy rules, expanding the dataset diversity, and integrating machine learning techniques for more precise predictions.
"
