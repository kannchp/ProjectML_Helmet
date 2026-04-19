# Motorcycle Helmet Detection & Safety Monitoring System

## 1. Project Overview

This project aims to develop an AI-based computer vision system capable of detecting motorcycles, identifying helmet usage, estimating approximate vehicle speed, and capturing safety violations such as multiple riders without helmets. The system is designed for deployment in university entrance areas to assist security personnel in monitoring road safety.

## 2. Objectives

1. Detect motorcycles in real-time video streams.
2. Detect helmet and no-helmet conditions for riders.
3. Count number of people on a motorcycle.
4. Estimate approximate vehicle speed.
5. Trigger event-based capture for violations (multiple riders + no helmet).
6. Generalize across different camera angles and environments.

## 3. Dataset Strategy

Training data is collected from multiple camera angles at the university entrance, combined with public traffic datasets for pretraining. Video clips are divided into separate time segments before frame extraction to avoid data leakage between training and validation sets.

- Train/Validation split performed at clip level, not frame level.
- Training set contains diverse lighting and traffic conditions.
- Validation includes unseen camera angles.
- 20% of frames contain no motorcycles to reduce false positives.

## 4. Detection Classes

The system uses three object detection classes to simplify training and improve robustness:

1. motorcycle
2. helmet (head wearing helmet)
3. no_helmet (head without helmet)

## 5. Model Architecture

The detection system is based on a YOLO object detection framework. A pretrained backbone is fine-tuned using university traffic footage to adapt to local environmental conditions. Post-processing logic is applied to associate detected heads with motorcycles.

## 6. Rider & Passenger Identification

Driver and passenger roles are not trained as separate classes. Instead, spatial relationships between detected heads and motorcycle bounding boxes are used. Relative position determines front (driver) and rear (passenger) placement.

## 7. Speed Estimation

Two optional speed estimation approaches are implemented:

1. Equation-based estimation when camera calibration data is available
2. A Random Forest regression model trained using motion features when calibration information is unknown

## 8. Event-Based Violation Detection

1. Motorcycle detected.
2. Two or more riders detected.
3. At least one rider without helmet.
4. System captures image and logs violation event.

## 9. Evaluation Method

- Validation set contains unseen camera environments.
- No augmentation applied to validation data.
- Performance measured using mAP, precision, recall, and false positive rate.

## 10. Expected Outcome

The final system is expected to operate robustly across different camera angles and lighting conditions, providing reliable helmet detection and safety monitoring suitable for real-world deployment at university entrances.
