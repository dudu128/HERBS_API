FROM python:3.9

COPY code /code
COPY requirements.txt /requirements.txt

RUN apt update -y
RUN apt upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN python -m pip install --upgrade pip
RUN python -m pip install --default-timeout=300 -r /requirements.txt

WORKDIR /code
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8001", "inference_herbs_api:app"]