# test.py
#
# Full prototype application:
#   - Person tracking + persistent Re-ID
#   - Weapon detection (stub mode until real model)
#   - Loitering detection
#   - Violence / fighting detection
#   - Central activity engine -> CSV + screenshot + console + HUD

import cv2
import time
from src.detection import PersonTracker
from src.weapon_detection import WeaponDetector
from src.loitering_detection import LoiteringDetector
from src.violence_detection import ViolenceDetector
from src.activity_engine import ActivityEngine
from src.ui import draw_hud


def main():
    print("=" * 60)
    print("AI CCTV -- Suspicious Activity Detection Prototype")
    print("=" * 60)

    tracker = PersonTracker(model_path="models/yolo11n.pt", use_reid=True)
    weapon_det = WeaponDetector(model_path="models/weapon_model.pt")
    loiter_det = LoiteringDetector(
        zone=(150, 100, 500, 400),
        min_duration_seconds=15,
        movement_threshold=10.0,
    )
    violence_det = ViolenceDetector(model_path="models/pose_model.pt")
    engine = ActivityEngine(cooldown_seconds=8.0)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam")
        return

    print("\nControls:")
    print("  Q = quit")
    print("  S = save snapshot of current frame")
    print("  C = clear recent alerts from HUD")
    print()

    fps_t = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Detections
        annotated, detections = tracker.process_frame(frame)

        weapon_events = weapon_det.update(frame, detections)
        weapon_det.draw(annotated, weapon_events)

        loiter_events = loiter_det.update(detections)
        loiter_det.draw(annotated, loiter_events)

        violence_events = violence_det.update(frame, detections)
        violence_det.draw(annotated, violence_events)

        # --- Engine: dedupe + log
        engine.process(
            frame=annotated,
            weapon_events=weapon_events,
            loiter_events=loiter_events,
            violence_events=violence_events,
        )

        # --- HUD
        draw_hud(annotated, {
            "persons": len(detections),
            "weapons": len(weapon_events),
        }, engine)

        # --- FPS
        now = time.time()
        dt = now - fps_t
        fps_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        cv2.putText(annotated, f"{fps:4.1f} FPS",
                    (annotated.shape[1] - 110, annotated.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("AI CCTV -- Prototype", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            from datetime import datetime
            fname = f"output/screenshots/snapshot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"[snapshot] {fname}")
        elif key == ord("c"):
            engine.recent_events.clear()
            print("[hud] cleared recent alerts")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal alerts logged: {engine.logger.total_alerts}")
    print(f"CSV: output/alerts.csv")


if __name__ == "__main__":
    main()
