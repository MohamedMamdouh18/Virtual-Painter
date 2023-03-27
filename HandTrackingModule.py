import cv2
import mediapipe as mp
import time


class HandDetector:
    """
      This class uses the Mediapipe library to detect and extract landmarks from hands in images or video streams.

      Attributes:
          mode (bool): A boolean value indicating whether to detect hands in real-time or from an image.
          maxHands (int): An integer value indicating the maximum number of hands to be detected.
          detectionCon (float): A float value between 0 and 1 representing the minimum confidence value required for
          hand detection.
          trackCon (float): A float value between 0 and 1 representing the minimum confidence value required for hand
          tracking.
          mpHands: The Hands module from the Mediapipe library.
          hands: The Hands object used for detecting and tracking hands.
          mpDraw: The DrawingUtils module from the Mediapipe library.

      Methods:
          __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5): Initializes a HandDetector object with
           specified parameters.
          processImage(self, img): Processes the hands in the input image with the Hand module.
          drawLandmarks(self, img, draw=True): Draws landmarks on the input image using the DrawingUtils module.
          getLandmarkPositions(self, img, handNo=0, draw=True): Returns a list of landmark positions for the specified
          hand in the input image.
    """

    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        """
               Initializes a HandDetector object.

               Args:
                   mode (bool): A boolean value indicating whether to detect hands in real-time or from an image.
                   maxHands (int): An integer value indicating the maximum number of hands to be detected.
                   detectionCon (float): A float value between 0 and 1 representing the minimum confidence value
                    required for hand detection.
                   trackCon (float): A float value between 0 and 1 representing the minimum confidence value required
                   for hand tracking.

               Returns:
                   None
        """

        # Initialize HandDetector with parameters
        self.landmarkList = None
        self.results = None
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionConfidence
        self.trackCon = trackConfidence

        # Create the Hands module from the Mediapipe library and set its parameters
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        # Create the DrawingUtils module from the Mediapipe library
        self.mpDraw = mp.solutions.drawing_utils

        # The fixed IDs for tip of each finger
        self.tipIds = [4, 8, 12, 16, 20]

    def processImage(self, img):
        """
              Processes the hands in the input image with the Hand module.

              Args:
                  img (numpy.ndarray): The input image to be processed.

              Returns:
                  None
        """

        # Convert image to RGB format
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the hands in the image with the Hand module
        self.results = self.hands.process(imgRGB)

    def drawLandmarks(self, img, draw=True):
        """
                Draws landmarks on the input image using the DrawingUtils module.

                Args:
                    img (numpy.ndarray): The input image to draw landmarks on.
                    draw (bool, optional): A boolean value indicating whether to draw landmarks on the input image.
                     Defaults to True.

                Returns:
                    numpy.ndarray: The image with landmarks drawn on it.
        """
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw landmarks on the image using the DrawingUtils module
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

    def getLandmarkPositions(self, img, handNo=0, draw=True):
        """
        Returns a list of landmark positions for the specified hand in the input image.

        Args:
            img (numpy.ndarray): The input image to extract landmark positions from.
            handNo (int, optional): The index of the hand to extract landmark positions for. Defaults to 0.
            draw (bool, optional): A boolean value indicating whether to draw landmarks on the input image.
             Defaults to True.

        Returns:
            list: A list containing the x,y coordinates of each landmark for the specified hand.
        """

        # Get the dimensions of the image
        height, width, channels = img.shape
        lmList = []
        if self.results.multi_hand_landmarks:
            # Get the latest hand on the screen
            myHand = self.results.multi_hand_landmarks[handNo]
            for landmarkID, lm in enumerate(myHand.landmark):
                # Get the x and y coordinates of each landmark and append them to a list
                cx, cy = int(lm.x * width), int(lm.y * height)
                lmList.append([landmarkID, cx, cy])
                if draw:
                    # Draw circles on the image at the location of each landmark
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), 2)

        self.landmarkList = lmList
        return lmList

    def fingersUp(self):
        """
           Determines if each finger is up or down based on landmark positions.

           Parameters
           ----------
           self : object
               An instance of a class that contains a list of landmark positions for each finger.

           Returns
           -------
           list
               A list containing binary values indicating whether each finger is up or not.

           Notes
           -----
           The function first initializes an empty list to store the state of each finger. It then checks the position of
           the thumb by comparing the x-coordinate of the topmost landmark of the thumb with the x-coordinate of the landmark
           below it. If the topmost landmark has a lower x-coordinate, the thumb is considered "up" and the value 1 is
           appended to the fingers list, otherwise 0 is appended -works only for right hand-.

           Next, the function iterates over the landmarks representing each finger (excluding the thumb) and checks if
           the y-coordinate of the topmost landmark is less than the y-coordinate of the landmark below it. If yes, then
           that finger is considered "up" and the value 1 is appended to the fingers list, else 0 is appended. At the end,
           the function returns the fingers list which contains binary values indicating whether each finger is up or not.
        """
        fingers = []
        # Thumb
        #left hand
        if self.landmarkList[self.tipIds[0]][1] < self.landmarkList[self.tipIds[4]][1] :
            if self.landmarkList[self.tipIds[0]][1] < self.landmarkList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        # right hand
        else:
            if self.landmarkList[self.tipIds[0]][1] > self.landmarkList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.landmarkList[self.tipIds[id]][2] < self.landmarkList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


def main():
    prevTime = 0
    # Initialize the video capture device
    cap = cv2.VideoCapture(0)
    # Create an instance of the HandDetector class
    detector = HandDetector()
    while True:
        # Capture a frame from the video feed
        success, img = cap.read()

        # Call the processImage method to detect the hands in the image
        detector.processImage(img)
        # Call the drawLandmarks method to draw landmarks on the image
        img = detector.drawLandmarks(img)
        # Call the getLandmarkPositions method to get the positions of the landmarks
        lmList = detector.getLandmarkPositions(img)

        if len(lmList) != 0:
            # Print the position of the landmark at index 4
            print(lmList[4])

        curTime = time.time()
        fps = 1 / (curTime - prevTime)
        prevTime = curTime

        # Put the FPS counter on the screen
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # Show the image on the screen
        cv2.imshow("Image", img)
        # Wait for a keyboard event
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
