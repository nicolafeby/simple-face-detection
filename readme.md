# 🎥 Face & Eye Detection with OpenCV  

This project uses **OpenCV** to detect faces and eyes in **real-time** via webcam using the **Haar Cascade Classifier**.  

## 🚀 Features  
- Real-time face detection via webcam.  
- Eye detection within the detected face area.  
- Draws **blue boxes** around faces and **green boxes** around eyes.  
- Labels faces with names like `Face_001`, `Face_002`, etc.  
- Press **q** to quit the application.  

## 📦 Requirements  
Make sure you have the following dependencies installed:  

```bash
pip install opencv-python
```

Run local server
```
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```
