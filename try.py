import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    ret , frame = cap.read()

    cv2.imshow("dfjlks", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()