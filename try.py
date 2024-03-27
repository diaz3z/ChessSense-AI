import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret , frame = cv2.imread()

    cv2.imshow("dfjlks", frame)