import cv2
import PIL.Image as Image
import os
import pandas as pd
import numpy as np


def capture_data(data_type, number_of_frames=0):
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    rval, frame = vc.read()
    data_path = os.path.join("../data", data_type)
    os.makedirs(data_path, exist_ok=True)
    indecies = [int(file.split("_")[-1].split(".")[0]) for file in os.listdir(data_path) if file.endswith(".jpg")]
    if indecies:
        i = np.max(indecies)
    else:
        i = 0
    name = []
    X = []
    Y = []
    W = []
    H = []

    number_of_frames = number_of_frames + i
    condition = True
    while condition:
      if frame is not None:
         cv2.imshow("preview", frame)

      rval, frame = vc.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # Detect the faces
      faces = face_cascade.detectMultiScale(
          gray,
          scaleFactor=1.1,
          minNeighbors=5,
          minSize=(80, 80),
      )

      # Draw the rectangle around each face
      if (len(faces) == 1 and data_type == "face") or (data_type != "face" and len(faces) == 0):
          i += 1
          im = Image.fromarray(frame)
          im = im.convert(mode='L')
          im.save(os.path.join(data_path, f"frame_{i}.jpg"))
          name.append(f"frame_{i}.jpg")
          if len(faces):
              (x, y, w, h) = faces[0]
              cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
              X.append(x)
              Y.append(y)
              W.append(w)
              H.append(h)

      if cv2.waitKey(1) & 0xFF == ord('q'):
         break

      condition = True if not number_of_frames else i <= number_of_frames

    if len(name) and data_type == "face":
        df = pd.DataFrame.from_dict({"name": name, "x": X, "y": Y, "w": W, "h": H})
        if data_type == "face" and "face.csv" in os.listdir(".."):
            old_df = pd.read_csv("../face.csv")
            old_df = old_df[["name", "x", "y", "w", "h"]]
            output_df = old_df.append(df, ignore_index=True)
            output_df.to_csv(f"face.csv")
        else:
            df.to_csv(f"face.csv")