FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

COPY download_model.py .
RUN python download_model.py

COPY handler.py .
COPY Airborne_1_beaded/ ./Airborne_1_beaded/
COPY Airborne_2/ ./Airborne_2/
COPY Airborne_1_PVC/ ./Airborne_1_PVC/
COPY Hercule/ ./Hercule/
COPY Plasma/ ./Plasma/

CMD ["python", "-u", "handler.py"]