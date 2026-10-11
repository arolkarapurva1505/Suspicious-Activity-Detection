import cv2
from ultralytics import YOLO

m = YOLO("models/pose_model.pt")
cap = cv2.VideoCapture(0)

print("Pose model loaded. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = m.predict(frame, verbose=False)
    annotated = results[0].plot()
    kp = results[0].keypoints
    n = len(kp) if kp is not None else 0
    cv2.putText(annotated, f"Poses: {n}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Pose Test", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
