FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies before copying the rest of the project to keep rebuilds fast.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Only copy the files needed to run the API in the container.
COPY app.py .
COPY model_utils.py .
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
