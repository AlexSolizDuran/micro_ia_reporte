# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias, incluyendo git-lfs para manejar archivos grandes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- NUEVO: Descargar el modelo desde el Hugging Face Hub ---
# Clona el repositorio del modelo usando Git LFS
RUN git lfs install
RUN git clone https://huggingface.co/ludicolo/modelo-v1/tree/main

# --- FIN DEL CAMBIO ---

# Copiar solo el código de Python (el modelo ya está descargado)
COPY main.py model.py ./

EXPOSE 8000

# El comando de inicio sigue igual
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]