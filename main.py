# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import generar_respuesta_ia

# --- DEFINICIÓN DE MODELOS DE DATOS (Pydantic) - Sin cambios ---
class PromptRequest(BaseModel):
    prompt: str

class SqlResponse(BaseModel):
    sql: str
    formato: str
    columnas: list[str] = []

# --- API FASTAPI ---
app = FastAPI(title="Text2SQL AI Service")

@app.post("/generar-sql", response_model=SqlResponse)
def generar_sql(request: PromptRequest):
    """
    Endpoint principal. Llama a la lógica del modelo y devuelve la respuesta.
    """
    try:
        # 1. Llama a la función que contiene toda la lógica de IA
        data_generada = generar_respuesta_ia(request.prompt)
        
        # 2. Devuelve los datos usando el modelo de respuesta de Pydantic
        return SqlResponse(**data_generada)
        
    except ValueError as e:
        # Captura los errores de lógica (ej. JSON inválido) y los devuelve como 500
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Captura cualquier otro error inesperado
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}