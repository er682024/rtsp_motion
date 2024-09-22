FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir flask opencv-python-headless numpy requests pytz pythonping

EXPOSE 5000

CMD ["python", "rtsp_motion.py"]
