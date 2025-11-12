FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia ambos archivos de Python y el modelo
COPY main.py model.py ./
COPY mi-modelo-entrenado /app/mi-modelo-entrenado

EXPOSE 8000

# El comando de inicio apunta a main.py
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]