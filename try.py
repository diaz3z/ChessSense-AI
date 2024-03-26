import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret ,frame = cap.read()
    cv2.imshow("ddkl", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.
cv2.destroyAllWindows()
