import requests
from PIL import Image, ImageFilter
import io
import numpy as np
import face_recognition
import time
import asyncio
from dotenv import load_dotenv
import os

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

class Rest:
    # Adaptive background (kept in memory across calls)
    _bg_small: np.ndarray | None = None

    async def detectMovement(image1, image2) -> float:
        """Return motion score as a changed-pixel ratio (0.0 ~ 1.0).

        Implements: downsample + grayscale + blur + thresholded change ratio
        against an adaptive background.

        Notes:
        - Keeps its own adaptive background; `image1`/`image2` are accepted to
          avoid changing the rest of your code. The first frame seen initializes
          the background.
        """
        # Tunables (keep these small and boring)
        target_size = (160, 90)   # (W, H) downsample for stability + speed
        blur_radius = 1.5         # blur to suppress MJPEG noise + tiny jitter
        diff_threshold = 20       # pixel intensity threshold (0-255)
        alpha = 0.02              # background update speed

        # Use the *current* frame (your call-site passes (empty_frame, now_frame))
        frame = image2

        # 1) Downsample + 2) Grayscale + 3) Blur
        pil = Image.fromarray(frame)
        pil = pil.resize(target_size, Image.BILINEAR).convert("L")
        pil = pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        cur = np.asarray(pil, dtype=np.float32)

        # Initialize background on first use
        if Rest._bg_small is None:
            Rest._bg_small = cur.copy()
            return 0.0

        bg = Rest._bg_small

        # 4) Motion as changed-pixel ratio against background
        diff = np.abs(cur - bg)
        changed = diff > float(diff_threshold)
        motion_ratio = float(np.mean(changed))  # 0.0 ~ 1.0

        # Adaptive background update:
        # - Update faster when scene is stable
        # - Update slower during motion so moving objects don't get absorbed instantly
        alpha_use = alpha if motion_ratio < 0.02 else (alpha * 0.2)
        Rest._bg_small = (1.0 - alpha_use) * bg + alpha_use * cur

        return motion_ratio

async def main():
    print("Prepearing face encoding...")
    my_portrait = face_recognition.load_image_file("my_portrait.jpeg")
    my_encoding = face_recognition.face_encodings(my_portrait, model="small")[0]
    print("Face encoding ready.")

    empty_frame = await Foundation.read_mjpeg_frame(URL, auth=(USER, PASS))
    while True:
        now_frame = await Foundation.read_mjpeg_frame(URL, auth=(USER, PASS))
        motion_ratio = await Rest.detectMovement(empty_frame, now_frame)
        print(motion_ratio)
        # if mse < 20:
        #     empty_frame = now_frame

    frame = await Foundation.read_mjpeg_frame(URL, auth=(USER, PASS))
    print(frame.shape)  # (H, W, 3)
    (success, SameFace) = await FaceRecognition.compare(frame, my_encoding)
    
    if success and SameFace:
        print("Pass.")
    elif success and not SameFace:
        print("Face detected, but does not match.")
    else:
        print("No face detected.")

if __name__ == "__main__":
    load_dotenv()

    # MotionEye
    ME_URL = os.getenv("ME_URL")
    USER = os.getenv("USER")
    PASS = os.getenv("PASS")

    # HomeAssistant
    HA_URL = os.getenv("HA_URL")
    TOKEN = os.getenv("TOKEN")
    
    asyncio.run(main())