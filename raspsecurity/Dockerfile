ARG BUILD_FROM
FROM $BUILD_FROM

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps:
# - dlib/face_recognition can be painful; this installs common build deps.
# - If you want less suffering, consider switching to a base that already supports dlib well.
RUN apk add --no-cache \
    python3 py3-pip \
    build-base cmake \
    jpeg-dev zlib-dev \
    libstdc++ \
    && python3 -m pip install --upgrade pip

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY main.py /app/main.py

# s6 overlay service files
COPY rootfs/ /

EXPOSE 3000