# ai_service.py

import json
import uvicorn
import torch
import sys 
from fastapi import FastAPI, HTTPException # <-- IMPORTACIÓN AÑADIDA
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

# --- CONFIGURACIÓN ---
CARPETA_MODELO_FINAL = "./mi-modelo-entrenado"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- DEFINICIÓN DE MODELOS DE DATOS (Pydantic) ---
class PromptRequest(BaseModel):
    prompt: str

class SqlResponse(BaseModel):
    sql: str
    formato: str
    columnas: list[str] = []


# --- CARGA DEL MODELO ---

print(f"Cargando modelo en: {DEVICE}")

try:
    tokenizer = T5Tokenizer.from_pretrained(CARPETA_MODELO_FINAL)
    model = T5ForConditionalGeneration.from_pretrained(CARPETA_MODELO_FINAL).to(DEVICE)
    model.eval() # <-- MEJORA 1: Pone el modelo en modo de evaluación para inferencia
except Exception as e:
    print(f"ERROR: No se pudo cargar el modelo o el tokenizador. Razón: {e}")
    # Usamos sys.exit para terminar la aplicación si el modelo no carga
    sys.exit(1)


# --- API FASTAPI ---

app = FastAPI(title="Text2SQL AI Service")

@app.post("/generar-sql", response_model=SqlResponse)
def generar_sql(request: PromptRequest):
    """
    Endpoint principal para recibir el prompt y generar el SQL/JSON.
    """
    prompt = request.prompt
    print("ESTE ES EL PROMPT " + prompt)
    # <-- MEJORA 2: Usa torch.no_grad() para desactivar el cálculo de gradientes (¡MÁS RÁPIDO!)
    with torch.no_grad():
        input_text = f"translate English to French: {prompt}"

        # 2. Tokenizar el prompt de entrada
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"].to(DEVICE)

        # 3. Generar la secuencia de salida (el JSON)
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=4,
            early_stopping=True,
            temperature=0.7
        )
        
        # 4. Decodificar la salida a texto
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("si llego aca")
    try:
        # 5. PARSEO SEGURO (AÑADE ESTA LÓGICA)
        
        # 5a. Limpiar la cadena de espacios extra y saltos de línea (opcional)
        processed_text = (
            generated_text
            .strip()
        )       
        # 5b. ¡AÑADIR LAS LLAVES FALTANTES!
        if not processed_text.startswith('{'):
            processed_text = '{' + processed_text
        if not processed_text.endswith('}'):
            processed_text = processed_text + '}'
            
        # 5c. Intentar el parseo con la cadena corregida
        data = json.loads(processed_text)
        print(data)
        # 6. Devolver el objeto
        return SqlResponse(**data)
        
    except json.JSONDecodeError as e:
        # <-- MEJORA 4: Captura errores de JSON y da más detalle
        print(f"Error al decodificar JSON: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"El modelo AI generó un JSON inválido. Salida cruda: {generated_text}"
        )
    except Exception as e:
        # <-- MEJORA 5: Captura CUALQUIER OTRO ERROR (como el ValidationError de Pydantic)
        print(f"Error inesperado al procesar la respuesta: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error al procesar la respuesta de la IA. Salida cruda: {generated_text}. Detalle del error: {str(e)}"
        )

# --- COMANDO DE EJECUCIÓN ---

if __name__ == "__main__":
    uvicorn.run("ai_service:app", host="0.0.0.0", port=8000, reload=False)