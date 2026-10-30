# LiveAttend: Face and Gesture Based Smart Attendance System

LiveAttend records attendance automatically using a webcam. It recognizes a person's face with deep learning, then asks for a random hand gesture (for example, "show 3 fingers") to confirm the person is live. Only when both checks pass is the attendance saved.

## Why this project?

- Manual attendance is slow and records can be lost or changed.
- Face-only systems can be fooled by holding up a photo or video.
- A random gesture cannot be answered by a photo, so it works as a simple liveness check.

## How it works

1. **Register:** face samples are captured and stored as 128-number embeddings in `encodings.pkl`.
2. **Recognize:** the live face is converted to an embedding and compared with the registered ones. A distance below 0.5 counts as a match.
3. **Challenge:** the system asks for a random number of fingers (1 to 5). MediaPipe Hands tracks 21 hand landmarks and counts the extended fingers.
4. **Verify and log:** if the correct gesture is held for 1 second, the name, date and time are saved to `attendance.csv`, once per person per day.

**Deep learning used**
- Face recognition: a ResNet model from the `face_recognition` (dlib) library turns each face into a 128-number embedding.
- Hand tracking: MediaPipe's neural networks detect the palm and predict 21 hand landmarks.

Both are pretrained models. This project combines them into one system and adds the liveness challenge and attendance logic.

## Finger counting

- Index, middle, ring and little finger: extended when the fingertip is farther from the wrist than its middle joint.
- Thumb: extended when the tip is farther from the base of the little finger than the thumb joint.
- It uses distances, not directions, so it works in any hand orientation.

## Project files

| File | Purpose |
|---|---|
| `register.py` | Registers a person and saves face embeddings |
| `attendance.py` | Runs recognition, the gesture challenge and attendance logging |
| `requirements.txt` | Required libraries |
| `encodings.pkl` | Created after registering (face data, keep private) |
| `attendance.csv` | Created after the first attendance (name, date, time) |


## Requirements

- Python 3.10 or 3.11
- A laptop or PC with a webcam
- Windows, macOS or Linux (tested workflow is Windows)

This needs a real camera window, so it does not run in Google Colab.

## Installation

Open a terminal inside the project folder, then run:

```
pip install -r requirements.txt
```

If `face_recognition` fails to install, install dlib first:

```
pip install dlib-bin
pip install face_recognition
```

## Usage

**1. Register a person**

```
python register.py "YourName"
```

- Click the camera window, then press **SPACE** 5 to 8 times, turning your face slightly each time.
- Press **Q** to finish.
- Run the command again with a different name to add more people.

**2. Run the attendance system**

```
python attendance.py
```

- Look at the camera. When you are recognized, your name appears in a green box.
- The screen shows "Show N finger(s)". Show that many fingers and hold them for 1 second.
- "attendance marked!" means it worked. The record is saved in `attendance.csv`.

**Keys in `attendance.py`**

| Key | Action |
|---|---|
| S | Save a screenshot to `demo/` |
| R | Start or stop recording `demo/demo.mp4` |
| Q | Quit |

## Settings

At the top of `attendance.py`:

| Setting | Default | Meaning |
|---|---|---|
| `MATCH_THRESHOLD` | 0.5 | Lower is stricter face matching |
| `CONFIRM_FRAMES` | 4 | Recognitions in a row before the challenge starts |
| `HOLD_SECONDS` | 1.0 | How long the correct gesture must be held |
| `CHALLENGE_TIMEOUT` | 10 | Seconds allowed to finish the gesture |
| `PROCESS_EVERY` | 3 | Run face recognition every N frames (higher is faster) |

## Removing data

- **Clear attendance:** delete `attendance.csv` (or delete one row in Excel). Close the file before running the program.
- **Remove everyone:** delete `encodings.pkl` and register again.
- **Remove one person:** save this as `remove_person.py`, then run `python remove_person.py "YourName"`:

```python
import pickle, sys

name = sys.argv[1]
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

keep = [i for i, n in enumerate(data["names"]) if n != name]
data["names"] = [data["names"][i] for i in keep]
data["encodings"] = [data["encodings"][i] for i in keep]

with open("encodings.pkl", "wb") as f:
    pickle.dump(data, f)
print("Removed", name)
```

Registering the same name again adds more samples, it does not replace the old ones.

## Troubleshooting

| Problem | Fix |
|---|---|
| `cv2.imshow` error: "function is not implemented" | Two OpenCV packages are clashing. Run `pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless`, then `pip install opencv-contrib-python==4.11.0.86` |
| Camera does not open | Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in both scripts |
| Says "already marked today" | Attendance is saved once per person per day. Delete `attendance.csv` to test again |
| "Unknown person" | Register again in better lighting with more samples |
| Slow or laggy | Increase `PROCESS_EVERY` (for example, 5) |
| `pkg_resources` warning | Harmless. Hide it with `pip install "setuptools<81"` |
| `'python' is not recognized` | Try `py` instead of `python` |

## Limitations

- Needs good lighting and handles one person at a time.
- Very similar faces may be confused.
- It can still be tricked if someone holds a registered person's photo and shows their own fingers. Blink detection or depth sensing would help.
- Accuracy has not been formally measured.

## Privacy

Only 128-number embeddings are stored, not face photos, and everything stays on the local computer. Do not share `encodings.pkl`. In real use, get people's consent.

## Future work

- Detect many students at once with YOLO
- Read ID cards with OCR
- Add blink detection against spoofing
- Streamlit dashboard for attendance reports
- Store records in a database such as SQLite