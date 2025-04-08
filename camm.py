import cv2

# Try different device indices if 0 doesn't work
cap = cv2.VideoCapture(2)  # Try /dev/video2 (use index 2)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if frame is read correctly
    if not ret:
        print("Can't receive frame")
        break
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()