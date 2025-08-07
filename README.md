Drowsiness Detection System:
This project is a real-time drowsiness detection system built using Python, OpenCV, and Dlib. It monitors a person's eyes using a webcam to detect signs of fatigue. If the system detects that the person's eyes have been closed for a certain period, it triggers an audible alarm to alert them. This can be a crucial safety tool for drivers, operators, or anyone who needs to remain alert.
Features
Real-time Monitoring: Processes live video feed from a webcam.

Facial Landmark Detection: Accurately identifies key facial features, specifically the eyes.

Eye Aspect Ratio (EAR) Calculation: Uses the EAR to determine the state of the eyes (open or closed).

Audible Alarm: Plays a sound to alert the user when drowsiness is detected.

Cross-platform: Runs on any operating system that supports Python and OpenCV.

Technologies & Libraries Used
Python 3.x

OpenCV: For video capture and image processing.

Dlib: For face detection and facial landmark prediction.

SciPy: For calculating the Euclidean distance between landmark points.

NumPy: For numerical operations on image arrays.


