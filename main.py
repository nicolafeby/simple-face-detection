import cv2

# Load classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Buka webcam (0 biasanya webcam default, bisa ubah ke 1/2 kalau ada kamera lain)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak bisa membuka kamera")
    exit()

i = 0
while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        i += 1
        cv2.putText(frame, ('Face_%03d' % i), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow('Webcam Face Detection', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()
