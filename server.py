from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import base64

app = FastAPI()

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.post("/detect")
async def detect_glasses(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        wearing_glasses = False

        # Proses deteksi wajah
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]

            # Deteksi "glasses" pakai thresholding di area mata
            edges = cv2.Canny(roi_gray, 100, 200)
            edge_density = np.sum(edges) / (w * h)

            if edge_density > 5:  # threshold sederhana
                wearing_glasses = True

            # Gambar rectangle di wajah
            color = (0, 255, 0) if wearing_glasses else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

        # Encode hasil deteksi ke base64
        _, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        return {
            "status_code": 200,
            "message": "Detection success",
            "data": {
                "wearing_glasses": wearing_glasses,
                "image": jpg_as_text
            }
        }
    except Exception as e:
        return {
            "status_code": 500,
            "message": f"Detection failed: {str(e)}",
            "data": None
        }
