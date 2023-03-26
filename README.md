# Virtual Painter App
This Python application allows you to draw on a canvas by tracking your hand gestures in real time using OpenCV and the MediaPipe Hands library. You can select different colors and brush sizes using your fingers, as well as clear the canvas and undo the last stroke.

## Table of Contents

- [Motivation](#Motivation)
- [Prerequisites](#Prerequisites)
- [Usages](#Usage)
- [Features](#Features)
- [Implementation Details](#Implementation-Details)
- [Sample Run](#Sample-Run)

## Motivation
The purpose of this project is to demonstrate how to use computer vision techniques to create an interactive drawing experience that doesn't require any hardware beyond a standard computer and webcam. This project showcases the capabilities of the MediaPipe Hands library for hand detection and gesture recognition, and demonstrates how to integrate it with OpenCV for real-time visual feedback.

## Prerequisites
This project requires Python 3.x and the following libraries:

- OpenCV
- NumPy
- MediaPipe

*You can install these libraries using pip:*

```
!pip install opencv-python numpy mediapipe
```

## Usage
To run the application, simply run the main.py script:
```
!python VirualPainter.py
```
This will open a new window showing the live camera feed with the drawing app overlayed on top.

## Features
The following features are available in the app:

- Brush size: Adjust the brush size by spreading or closing your index and middle fingers.
- Color selection: Select a color by extending your thumb and selecting one of four colors displayed at the top of the screen.
- Undo: Undo the last stroke by making a fist.
- Clear canvas: Clear the entire canvas by extending all fingers.

## Implementation Details
The application uses the MediaPipe Hands library to detect hand landmarks and recognize hand gestures. For each frame of video captured by the webcam, the landmarks are detected and analyzed to determine which gestures the user is making. The application then responds to these gestures by updating the state of the drawing canvas and overlaying it on top of the live video feed.

The drawing canvas itself is implemented using OpenCV, with each stroke added to a separate image that is merged with the live video feed when displayed on screen. Brush size and color selection are also handled using OpenCV functions for drawing shapes and filling regions of an image.

## Sample Run
![image](https://drive.google.com/uc?export=view&id=1tAizDqBhuWzbtOjbFd-2ZX6C9yS5jC24)
