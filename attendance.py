#attendance captures a person's face and verifies their identity using face recognition and a hand gesture challenge.
#If the person is recognized and successfully completes the gesture challenge, their attendance is marked in a CSV file.

import cv2, time, random, pickle, csv, os
from datetime import datetime
import numpy as np
import face_recognition
import mediapipe as mp

ENC_FILE, LOG_FILE = "encodings.pkl", "attendance.csv"
MATCH_THRESHOLD = 0.5      # lower = stricter
CONFIRM_FRAMES = 4         # consecutive recognitions before challenge starts
HOLD_SECONDS = 1.0         # how long the correct gesture must be held
CHALLENGE_TIMEOUT = 10     # seconds to complete the gesture
PROCESS_EVERY = 3          # run face recognition every N frames (speed)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def count_fingers(lm):
    """Count extended fingers from 21 MediaPipe hand landmarks.
    Orientation-independent: compares distances from the wrist."""
    p = lambda i: np.array([lm[i].x, lm[i].y])
    wrist = p(0)
    count = 0
    # index, middle, ring, pinky: tip farther from wrist than the PIP joint
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if np.linalg.norm(p(tip) - wrist) > np.linalg.norm(p(pip) - wrist) * 1.1:
            count += 1
    # thumb: tip farther from pinky base than the thumb IP joint
    if np.linalg.norm(p(4) - p(17)) > np.linalg.norm(p(3) - p(17)):
        count += 1
    return count


def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    rows = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, newline="") as f:
            rows = list(csv.reader(f))
    if any(r[0] == name and r[1] == today for r in rows[1:]):
        return False
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["name", "date", "time"])
        w.writerow([name, today, now])
    return True


def identify(rgb_small, known_enc, known_names):
    boxes = face_recognition.face_locations(rgb_small)
    if len(boxes) != 1:
        return None, boxes
    enc = face_recognition.face_encodings(rgb_small, boxes)[0]
    dists = face_recognition.face_distance(known_enc, enc)
    i = int(np.argmin(dists))
    name = known_names[i] if dists[i] < MATCH_THRESHOLD else "Unknown"
    return name, boxes


def put(img, text, y, color=(255, 255, 255)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main():
    if not os.path.exists(ENC_FILE):
        print("No registered faces. Run: python register.py \"Name\"")
        return
    with open(ENC_FILE, "rb") as f:
        data = pickle.load(f)
    known_enc, known_names = data["encodings"], data["names"]

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7,
                           min_tracking_confidence=0.6)

    state, who, streak = "SEARCH", None, 0
    target, deadline, hold_start = 0, 0, None
    message, message_until = "", 0
    boxes_draw, frame_i = [], 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame_i += 1
        now = time.time()

        # ---- face recognition (every few frames, on a downscaled image)
        current = None
        if frame_i % PROCESS_EVERY == 0:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            current, boxes = identify(cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
                                      known_enc, known_names)
            boxes_draw = [(t * 4, r * 4, b * 4, l * 4) for (t, r, b, l) in boxes]
            face_checked = True
        else:
            face_checked = False

        if state == "SEARCH":
            put(frame, "Look at the camera...", 30)
            if face_checked:
                if current and current != "Unknown":
                    streak = streak + 1 if current == who else 1
                    who = current
                else:
                    streak, who = 0, None
                    if current == "Unknown":
                        message, message_until = "Unknown person", now + 1
            if streak >= CONFIRM_FRAMES:
                target = random.randint(1, 5)
                deadline, hold_start = now + CHALLENGE_TIMEOUT, None
                state = "CHALLENGE"

        elif state == "CHALLENGE":
            # the person in front of the camera must stay the same
            if face_checked and current != who:
                state, streak, who = "SEARCH", 0, None
                message, message_until = "Face changed - restart", now + 2
            else:
                res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                fingers = None
                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                    fingers = count_fingers(lm.landmark)
                put(frame, f"Hi {who}! Show {target} finger(s)", 30, (0, 255, 255))
                put(frame, f"Detected: {fingers if fingers is not None else '-'}   "
                           f"Time left: {int(deadline - now)}s", 65)
                if fingers == target:
                    hold_start = hold_start or now
                    if now - hold_start >= HOLD_SECONDS:
                        new = mark_attendance(who)
                        message = (f"{who}: attendance marked!" if new
                                   else f"{who}: already marked today")
                        message_until = now + 4
                        state, streak = "DONE", 0
                else:
                    hold_start = None
                if now > deadline and state == "CHALLENGE":
                    message, message_until = "Verification failed", now + 3
                    state, streak, who = "SEARCH", 0, None

        elif state == "DONE":
            if now > message_until:
                state, who, streak = "SEARCH", None, 0

        # ---- drawing
        for (t, r, b, l) in boxes_draw:
            col = (0, 255, 0) if who else (0, 0, 255)
            cv2.rectangle(frame, (l, t), (r, b), col, 2)
            if who:
                cv2.putText(frame, who, (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        if now < message_until:
            put(frame, message, frame.shape[0] - 20, (0, 255, 0))

        cv2.imshow("Smart Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
