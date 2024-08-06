# AI-Posture-Estimation

Required Libraries

Mediapipe

It is a cross-platform library developed by Google that provides amazing ready-to-use ML solutions for computer vision tasks.

OpenCV 

A library in python is a computer vision library that is widely used for image analysis, image processing, detection, recognition, etc.

Installing required libraries

pip install opencv-python mediapipe msvc-runtime

![Screenshot (10)](https://user-images.githubusercontent.com/114305058/214681395-0120e3ad-f3c0-42a1-9786-d6e393a21d56.png)




Used 4 different angles to detect the squat:

1. Elbow joint (A), Shoulder joint(B), Hip joint(C) and Knee joint(D).

2. Apply the three points method to calculate the angle between two lines.

3. See if the value for the angle between Hip join (C) and Knee joint (D) is less than 130 degrees.

4. To determine the correct hand position for squat I’ve set the value of the Elbow joint (A) to greater than 130 degrees, and the value of the Shoulder joint(B) to greater than 30 degrees and less than 120 degrees.

5. The mean square error is used to train the model.

![Screenshot (7)](https://user-images.githubusercontent.com/114305058/214681221-45f1e495-995e-475f-980e-c563b4d1000e.png)
![Screenshot (8)](https://user-images.githubusercontent.com/114305058/214681271-4551cf57-11e4-4cf1-b354-6ed4dcbcaa92.png)

