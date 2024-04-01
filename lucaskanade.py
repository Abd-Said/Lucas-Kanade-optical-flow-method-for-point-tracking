import cv2
import numpy as np

# Mouse callback function
def select_point(event, x, y, flags, params):
    global point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        old_points = np.array([[x, y]], dtype=np.float32)
        point_selected = True

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

point_selected = False
old_points = np.array([])

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)

while True:
    ret, frame = cap.read()
    mean_val = np.mean(frame[:, :, 0])
    print(mean_val)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if point_selected is True:
       
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None)
        
        x, y = new_points.ravel()
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        for i, (new, old) in enumerate(zip(new_points, old_points)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)

        old_points = new_points
        old_gray = gray_frame

    cv2.imshow("Frame", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
