import cv2
import numpy as np
from ultralytics import YOLO
import torch


# Load the trained YOLOv8 model

torch.cuda.set_device(0)  # Set to your desired GPU number
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO('runs/detect/train2/weights/best.pt')
model.to(device)

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def detect_chess_board(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny algorithm
    edges = cv2.Canny(blur, 50, 150)

    # Find contours in the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for chess board dimensions
    board_contour = None
    board_area = 0

    # Loop through the contours to find the largest quadrilateral (the chess board)
    for cnt in contours:
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

        # Check if the approximated contour has 4 vertices (quadrilateral)
        if len(approx) == 4:
            # Calculate the area of the contour
            area = cv2.contourArea(cnt)

            # Update the board contour and area if the current contour is larger
            if area > board_area:
                board_contour = approx
                board_area = area

    return board_contour

def warp_chess_board(frame, board_contour, new_width, new_height):
    # Order the points of the chess board contour
    ordered_points = reorder(board_contour)

    # Compute the perspective transform matrix
    pts1 = np.float32(ordered_points)
    pts2 = np.float32([[0, 0], [new_width, 0], [0, new_height], [new_width, new_height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Apply the perspective transform to the frame
    warped = cv2.warpPerspective(frame, matrix, (new_width, new_height))

    # Detect and localize chess pieces on the warped image
    results = model(warped, stream=True)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box.xyxy[0].tolist()
            cv2.rectangle(warped, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    return warped

def localize_squares(warped, new_width, new_height):
    # Localize the squares on the warped chess board
    square_size = min(new_width // 8, new_height // 8)  # Adjust square size based on smaller dimension
    board_width = square_size * 8
    board_height = square_size * 8

    # Calculate the starting position for the chess board
    start_x = (new_width - board_width) // 2
    start_y = (new_height - board_height) // 2

    for i in range(8):
        for j in range(8):
            x = start_x + j * square_size
            y = start_y + i * square_size
            cv2.rectangle(warped, (x, y), (x + square_size, y + square_size), (0, 255, 0), 2)

    # Detect and localize chess pieces on the localized squares
    results = model(warped, stream=True)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box.xyxy[0].tolist()
            cv2.rectangle(warped, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    return warped

# Load the video
cap = cv2.VideoCapture('Video/1.mp4')

# Get the original video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the desired resolution
new_width = 640
new_height = 480

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to the desired resolution
    frame = cv2.resize(frame, (new_width, new_height))

    # Detect the chess board contour
    board_contour = detect_chess_board(frame)

    # Draw the detected chess board on the original frame
    if board_contour is not None:
        cv2.drawContours(frame, [board_contour], 0, (0, 255, 0), 2)

        # Warp the chess board to a top-down view and localize pieces
        warped = warp_chess_board(frame, board_contour, new_width, new_height)

        # Localize the squares and pieces on the warped image
        warped = localize_squares(warped, new_width, new_height)

        # Display the original frame with the detected chess board
        cv2.imshow('Chess Board Detection', frame)

        # Display the warped chess board with localized squares and pieces
        cv2.imshow('Warped Chess Board', warped)
    else:
        cv2.imshow('Chess Board Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()