# test_weapon.py
import cv2
from src.weapon_detection import WeaponDetector

wd = WeaponDetector(model_path="models/weapon_model.pt")

cap = cv2.VideoCapture(0)
print("Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # No person detections yet — just run the weapon detector alone
    events = wd.update(frame=frame, person_detections=[])
    frame = wd.draw(frame, events)

    cv2.imshow("Weapon Detector (stub mode)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()