import cv2
import numpy as np

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

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# Convert real-life dimensions to pixels based on the resolution of the image
# Assuming the resolution of the image is known


def filter_chessboard_contours(contours):
    filtered_contours = []
    for cnt in contours:
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        # Check if the approximated contour has 4 vertices (quadrilateral)
        if len(approx) == 4:
            # Calculate the area of the contour
            area = cv2.contourArea(cnt)

            # Check if the area of the contour matches the expected chessboard size
            if abs(area - chessboard_width_px * chessboard_height_px) < 500:  # Adjust threshold as needed
                filtered_contours.append(approx)

    return filtered_contours
def detect_chess_board(frame):
    thresh = preprocess_frame(frame)
    # Apply morphological operations to clean up the image
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and aspect ratio
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 1000 < area < 30000:  # Adjust the area thresholds as needed
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                filtered_contours.append(approx)

    # Sort contours based on area
    filtered_contours.sort(key=cv2.contourArea, reverse=True)

    # Return the largest contour
    if filtered_contours:
        return filtered_contours[0]
    else:
        return None


def warp_chess_board(frame, board_contour, new_width, new_height):
    # Order the points of the chess board contour
    ordered_points = reorder(board_contour)

    # Compute the perspective transform matrix
    pts1 = np.float32(ordered_points)
    pts2 = np.float32([[0, 0], [new_width, 0], [0, new_height], [new_width, new_height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Apply the perspective transform to the frame
    warped = cv2.warpPerspective(frame, matrix, (new_width, new_height))

    return warped

def localize_squares(warped, square_size, grid_width, grid_height):
    # Localize the squares on the warped chess board
    # Adjust square size based on smaller dimension
    board_width = grid_width
    board_height = grid_height

    # Calculate the starting position for the chess board
    start_x = (new_width - board_width) // 2
    start_y = (new_height - board_height) // 2

    for i in range(8):
        for j in range(8):
            x = start_x + j * square_size
            y = start_y + i * square_size
            cv2.rectangle(warped, (x, y),
                          (x + square_size, y + square_size),
                          (0, 255, 0), 2)

    return warped

# Load the video
cap = cv2.VideoCapture(0)

# Get the original video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

square_size = 70
# Define the desired resolution
new_width = 650
new_height = 600

grid_width = square_size * 8
grid_height = square_size * 8

# Define the real-life dimensions of the chessboard
chessboard_width_cm = 25  # in centimeters
chessboard_height_cm = 25  # in centimeters

chessboard_width_px = int(chessboard_width_cm * (width / new_width))
chessboard_height_px = int(chessboard_height_cm * (height / new_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (new_width, new_height))

    # Detect the chess board contour
    board_contour = detect_chess_board(frame)

    # Draw the detected chess board on the original frame
    if board_contour is not None:
        cv2.drawContours(frame, [board_contour], 0, (0, 255, 0), 2)

        # Warp the chess board to a top-down view and localize pieces
        warped = warp_chess_board(frame, board_contour, new_width, new_height)

        # Localize the squares on the warped image
        warped = localize_squares(warped, square_size, grid_width, grid_height)

        # Display the original frame with the detected chess board
        cv2.imshow('Chess Board Detection', frame)

        # Display the warped chess board with localized squares
        cv2.imshow('Warped Chess Board', warped)
    else:
        cv2.imshow('Chess Board Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
