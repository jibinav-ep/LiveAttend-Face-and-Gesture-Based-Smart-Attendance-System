#register the face of a person in front of the camera, and save the encodings to encodings.pkl

import sys, os, pickle
import cv2
import face_recognition

ENC_FILE = "encodings.pkl"

def load():
    if os.path.exists(ENC_FILE):
        with open(ENC_FILE, "rb") as f:
            return pickle.load(f)
    return {"names": [], "encodings": []}

def main():
    if len(sys.argv) < 2:
        print('Usage: python register.py "Name"')
        return
    name = sys.argv[1]
    data = load()
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        view = frame.copy()
        for (t, r, b, l) in boxes:
            cv2.rectangle(view, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(view, f"{name}: {count} samples | SPACE=capture  Q=done",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Register", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if len(boxes) != 1:
                print("Need exactly one face in the frame.")
                continue
            enc = face_recognition.face_encodings(rgb, boxes)[0]
            data["names"].append(name)
            data["encodings"].append(enc)
            count += 1
            print(f"Captured sample {count}")
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    with open(ENC_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {count} samples for {name}.")

if __name__ == "__main__":
    main()
