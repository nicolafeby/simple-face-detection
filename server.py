from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import os

app = FastAPI(title="Face & Eye Detector API")

# Load Haar cascades
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"

if not os.path.exists(FACE_CASCADE_PATH) or not os.path.exists(EYE_CASCADE_PATH):
    raise RuntimeError("Haarcascade file tidak ditemukan di cv2.data.haarcascades")

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)


def encode_image_to_base64(img_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", img_bgr)
    if not ok:
        raise RuntimeError("Gagal encode gambar")
    return base64.b64encode(buffer).decode("utf-8")


@app.post("/detect")
async def detect_faces_eyes(file: UploadFile = File(...), return_image: bool = True):
    """
    Request:
      - multipart/form-data, field 'file' (jpg/png)
    Response:
      {
        "status_code": int,
        "message": str,
        "data": {
          "faces_count": int,
          "eyes_count": int,
          "boxes": {
            "faces": [[x,y,w,h], ...],
            "eyes": [[x,y,w,h], ...]
          },
          "face_detected": bool,
          "image": base64 or null
        }
      }
    """
    try:
        # Validasi format file
        if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
            return JSONResponse(
                status_code=400,
                content={"status_code": 400, "message": "Unsupported file type", "data": None},
            )

        contents = await file.read()
        if len(contents) == 0:
            return JSONResponse(
                status_code=400,
                content={"status_code": 400, "message": "Empty file", "data": None},
            )

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(
                status_code=400,
                content={"status_code": 400, "message": "Failed to decode image", "data": None},
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        faces = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in faces]

        eyes_all = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # Deteksi mata di setiap wajah
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
            for (ex, ey, ew, eh) in eyes:
                eyes_all.append([int(x+ex), int(y+ey), int(ew), int(eh)])
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

            # Gambar bounding box wajah
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        response_data = {
            "faces_count": len(faces),
            "eyes_count": len(eyes_all),
            "boxes": {"faces": faces, "eyes": eyes_all},
            "face_detected": len(faces) > 0,
            "image": encode_image_to_base64(img) if return_image else None,
        }

        status_code = 200 if len(faces) > 0 else 404
        message = "Detection success" if len(faces) > 0 else "No face detected"

        return JSONResponse(
            status_code=status_code,
            content={
                "status_code": status_code,
                "message": message,
                "data": response_data,
            },
        )

    except Exception as e:
        print("Error in /detect:", str(e))
        return JSONResponse(
            status_code=500,
            content={
                "status_code": 500,
                "message": f"Detection failed: {str(e)}",
                "data": None,
            },
        )
