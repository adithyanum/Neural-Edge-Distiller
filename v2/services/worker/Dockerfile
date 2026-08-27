FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY worker.py config.py ./
COPY services/ services/
COPY kaggle_scripts/ kaggle_scripts/
CMD ["python", "-u", "worker.py"]