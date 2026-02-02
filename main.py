import requests
from PIL import Image
import io
import numpy as np
import face_recognition
import time

def read_mjpeg_frame(url: str, auth, timeout=10):
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

def compare(frame, my_encoding) -> (bool, bool): # first for if got the face, second for comparism
    try:
        unknown_encoding = face_recognition.face_encodings(frame, model="small")[0]
    except:
        print("Unexpected error as encoding")
        return(False, False)
    
    result = face_recognition.compare_faces([my_encoding], unknown_encoding)
    return(True, result)

def main():
    my_portrait = face_recognition.load_image_file("my_portrait.jpeg")
    my_encoding = face_recognition.face_encodings(my_portrait, model="small")[0]

    StartTime = time.time()

    frame = read_mjpeg_frame(URL, auth=(USER, PASS))
    print(frame.shape)  # (H, W, 3)
    (sucess, SameFace) = compare(frame, my_encoding)

    if SameFace:
        print("abcabc")
    
    EndTime = time.time()
    print("One cycle time: ", EndTime - StartTime)


if __name__ == "__main__":
    URL = "http://192.168.31.6:8375"
    USER = "user"
    PASS = "UserPassword"

    main()