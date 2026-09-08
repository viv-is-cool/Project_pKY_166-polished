FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt ./
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
COPY backend/ ./
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
