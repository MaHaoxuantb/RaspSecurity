import requests
from PIL import Image, ImageFilter
import io
import numpy as np
import face_recognition
import time
import asyncio
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import uvicorn

# from Deprecated.rest import Rest # deprecated
# from Deprecated.trigger import trigger # deprecated

class Foundation:
    async def read_mjpeg_frame(url: str, auth, timeout=10):
        r = requests.get(url, auth=auth, stream=True, timeout=timeout)
        r.raise_for_status()

        buf = bytearray()
        for chunk in r.iter_content(chunk_size=4096):
            if not chunk:
                continue
            buf += chunk

            a = buf.find(b"\xff\xd8")  # JPEG start
            b = buf.find(b"\xff\xd9")  # JPEG end
            if a != -1 and b != -1 and b > a:
                jpg = bytes(buf[a:b+2])
                del buf[:b+2]
                img = Image.open(io.BytesIO(jpg)).convert("RGB")
                return np.array(img)

        raise RuntimeError("No JPEG frame found in stream")


class FaceRecognition:
    async def compare(frame, my_encoding) -> (bool, bool): # first for if got the face, second for comparism
        StartTime = time.time()
        try:
            unknown_encodings = face_recognition.face_encodings(frame, model="small")
            if not unknown_encodings:
                print("No face found in frame")
                return (False, False)
            unknown_encoding = unknown_encodings[0]
        except Exception as e:
            print(f"Unexpected error while encoding face: {e}")
            return (False, False)

        result = face_recognition.compare_faces([my_encoding], unknown_encoding)

        EndTime = time.time()
        print("FacialRecognition Time: ", EndTime - StartTime)

        return (True, bool(result[0]))
    
    async def prepare():
        print("Prepearing face encoding...")
        my_portrait = face_recognition.load_image_file("my_portrait.jpeg")
        my_encoding = face_recognition.face_encodings(my_portrait, model="small")[0]
        print("Face encoding ready.")
        return my_encoding
    
class Database:
    async def prepare():
        conn = sqlite3.connect("security.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                service TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    async def log_event(timestamp, service, event_type, details=None):
        conn = sqlite3.connect("security.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO events (timestamp, service, event_type, details)
            VALUES (?, ?, ?, ?)
        """, (timestamp, service, event_type, details))
        conn.commit()
        conn.close()


async def main():
    # empty_frame = await Foundation.read_mjpeg_frame(URL, auth=(USER, PASS))
    # while True:
    #     now_frame = await Foundation.read_mjpeg_frame(URL, auth=(USER, PASS))
    #     motion_ratio = await Rest.detectMovement(empty_frame, now_frame)
    #     print(motion_ratio)
    #     # if mse < 20:
    #     #     empty_frame = now_frame

    frame = await Foundation.read_mjpeg_frame(ME_URL, auth=(USER, PASS))
    print(frame.shape)  # (H, W, 3)
    (success, SameFace) = await FaceRecognition.compare(frame, my_encoding)
    if success and SameFace:
        print("Pass.")
    elif success and not SameFace:
        print("Face detected, but does not match.")
    else:
        print("No face detected.")


app = FastAPI()

# Expected JSON body: {"security_key": "...", "entity": "Cam1", "trigger": true}
class TriggerRequest(BaseModel):
    security_key: str
    entity: str
    trigger: bool

@app.post("/recognize_current")
async def recognize_current(request: TriggerRequest):
    # Request 
    if request.security_key != SECURITY_KEY:
        print("Not Valid Security Key Attempted: ", request.security_key)
        return {"status": "error", "message": "Invalid security key."}
    if request.trigger: # if true
        if request.entity == "Cam1":
            asyncio.create_task(Database.log_event(time.strftime("%Y-%m-%d %H:%M:%S"), "TriggerRequest", "Success", "Triggerred by Cam1"))
        elif request.entity == "UltraSonic":
            asyncio.create_task(Database.log_event(time.strftime("%Y-%m-%d %H:%M:%S"), "TriggerRequest", "Success", "Triggerred by UltraSonic"))
        else:
            asyncio.create_task(Database.log_event(time.strftime("%Y-%m-%d %H:%M:%S"), "TriggerRequest", "Error", f"Unknown entity: {request.entity}"))
            return {"status": "error", "message": "WTF are to trying to do?"}
    else: 
        asyncio.create_task(Database.log_event(time.strftime("%Y-%m-%d %H:%M:%S"), "TriggerRequest", "Error", f"Unknown entity: {request.entity}"))
        return {"status": "error", "message": "WTF are to trying to do?"}
    # Compare
    try:
        frame = await Foundation.read_mjpeg_frame(ME_URL, auth=(USER, PASS))
        got_face, compare_success = await FaceRecognition.compare(frame, my_encoding)
        if not got_face:
            asyncio.create_task(Database.log_event(time.strftime("%Y-%m-%d %H:%M:%S"), "FacialRecognition", "UnSuccess", "RecognitionAttempt: No face detected"))
            return {"status": "unsuccess"}
        else:
            asyncio.create_task(Database.log_event(time.strftime("%Y-%m-%d %H:%M:%S"), "FacialRecognition", "Success", f"RecognitionAttempt: Face detected, compare successful?: {compare_success}"))
            return {"status": "success", "match": compare_success}
    except Exception as e:
        print(f"Error during recognition: {e}")
        asyncio.create_task(Database.log_event(time.strftime("%Y-%m-%d %H:%M:%S"), "FacialRecognition", "Error", str(e)))
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    load_dotenv()

    # MotionEye
    ME_URL = os.getenv("ME_URL")
    USER = os.getenv("USER")
    PASS = os.getenv("PASS")

    # HomeAssistant
    HA_URL = os.getenv("HA_URL")
    TOKEN = os.getenv("TOKEN")

    # Security
    SECURITY_KEY = os.getenv("security_key")

    my_encoding = asyncio.run(FaceRecognition.prepare())
    
    asyncio.run(Database.prepare())

    # asyncio.run(main())
    uvicorn.run(app, host="0.0.0.0", port=3000)