#!/usr/bin/env python3
"""
collect_currency_photos.py — Collect banknote photos for YOLO training
=======================================================================
Uses your Dell laptop webcam to photograph banknotes.
Press SPACE to capture a photo, Q to quit.

Usage:
    python collect_currency_photos.py

Output:
    data/currency_photos/10/photo_001.jpg
    data/currency_photos/20/photo_001.jpg
    ... etc.

After collecting photos, upload them to Roboflow for labelling.
"""

import cv2
import os
import sys
import time

DENOMINATIONS = ["10", "20", "50", "100", "200", "500", "2000"]
OUTPUT_DIR    = "data/currency_photos"

def main():
    print("=" * 50)
    print("  Currency Photo Collector")
    print("  SPACE = capture | Q = next note | ESC = quit")
    print("=" * 50)

    # Create output folders
    for denom in DENOMINATIONS:
        os.makedirs(os.path.join(OUTPUT_DIR, denom), exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # CAP_DSHOW for Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check CAMERA_INDEX.")
        sys.exit(1)

    for denom_idx, denom in enumerate(DENOMINATIONS):
        photo_dir   = os.path.join(OUTPUT_DIR, denom)
        count       = len([f for f in os.listdir(photo_dir) if f.endswith('.jpg')])
        target      = 80   # aim for 80 photos per denomination
        print(f"\n[{denom_idx+1}/{len(DENOMINATIONS)}] Now collecting: ₹{denom} note")
        print(f"  Already have: {count} photos | Target: {target}")
        print(f"  Hold the ₹{denom} note in front of the camera.")
        print(f"  Vary the angle, distance, and lighting.")
        print(f"  SPACE = capture | Q = move to next denomination")

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            display = frame.copy()

            # HUD overlay
            cv2.putText(display, f"Denomination: Rs.{denom}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Photos: {count}/{target}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(display, "SPACE=capture  Q=next  ESC=quit",
                        (10, display.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (180, 180, 180), 1)

            # Progress bar
            bar_w   = int((count / target) * (display.shape[1] - 20))
            bar_col = (0, 200, 100) if count >= target else (0, 150, 255)
            cv2.rectangle(display, (10, 80), (display.shape[1]-10, 95),
                          (60, 60, 60), -1)
            if bar_w > 0:
                cv2.rectangle(display, (10, 80), (10 + bar_w, 95), bar_col, -1)

            if count >= target:
                cv2.putText(display, "TARGET REACHED! Press Q for next.",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 100), 2)

            cv2.imshow("Currency Photo Collector", display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("\n[Collector] Quitting.")
                cap.release()
                cv2.destroyAllWindows()
                _print_summary()
                sys.exit(0)

            elif key == ord('q') or key == ord('Q'):
                print(f"[Collector] Done with ₹{denom}. Photos saved: {count}")
                break

            elif key == ord(' '):
                count += 1
                filename = os.path.join(photo_dir, f"photo_{count:03d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"  Saved: {filename}")
                # Flash effect
                flash = display.copy()
                cv2.rectangle(flash, (0,0), (display.shape[1], display.shape[0]),
                              (255,255,255), -1)
                cv2.addWeighted(flash, 0.3, display, 0.7, 0, display)
                cv2.imshow("Currency Photo Collector", display)
                cv2.waitKey(80)

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Collector] All denominations collected!")
    _print_summary()

def _print_summary():
    print("\n" + "=" * 50)
    print("  Photo collection summary:")
    total = 0
    for denom in DENOMINATIONS:
        photo_dir = os.path.join(OUTPUT_DIR, denom)
        if os.path.exists(photo_dir):
            n = len([f for f in os.listdir(photo_dir) if f.endswith('.jpg')])
            total += n
            status = "OK" if n >= 50 else f"need {50-n} more"
            print(f"  Rs.{denom:>4}: {n:>3} photos  [{status}]")
    print(f"\n  Total photos: {total}")
    print(f"\n  Next step:")
    print(f"  1. Go to https://roboflow.com")
    print(f"  2. Create project → Object Detection")
    print(f"  3. Upload photos from: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  4. Label each note with its denomination")
    print(f"  5. Export as YOLOv8 format")
    print("=" * 50)

if __name__ == "__main__":
    main()