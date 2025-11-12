# model.py
import json
import torch
import sys
from transformers import T5ForConditionalGeneration, T5Tokenizer

# --- CONFIGURACIÓN (Sin cambios) ---
CARPETA_MODELO_FINAL = "./modelo-v1/mi-modelo-entrenado"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- CARGA DEL MODELO (Sin cambios) ---
print(f"Cargando modelo en: {DEVICE}")

try:
    tokenizer = T5Tokenizer.from_pretrained(CARPETA_MODELO_FINAL)
    model = T5ForConditionalGeneration.from_pretrained(CARPETA_MODELO_FINAL).to(DEVICE)
    model.eval()
except Exception as e:
    print(f"ERROR: No se pudo cargar el modelo o el tokenizador. Razón: {e}")
    sys.exit(1)

# --- FUNCIÓN DE GENERACIÓN (Contiene toda tu lógica original) ---
def generar_respuesta_ia(prompt: str):
    """
    Ejecuta la lógica completa: recibe un prompt, usa el modelo y devuelve el diccionario parseado.
    """
    # Lógica original sin modificar
    print("ESTE ES EL PROMPT " + prompt)
    with torch.no_grad():
        # NOTA: Se mantiene la lógica original del prompt.
        input_text = f"translate English to French: {prompt}"

        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"].to(DEVICE)

        # NOTA: Se mantiene el parámetro 'temperature' de tu código original.
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=4,
            early_stopping=True,
            temperature=0.7
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("si llego aca")
        
    try:
        # Lógica de parseo original sin modificar
        processed_text = generated_text.strip()
        
        if not processed_text.startswith('{'):
            processed_text = '{' + processed_text
        if not processed_text.endswith('}'):
            processed_text = processed_text + '}'
            
        data = json.loads(processed_text)
        print(data)
        
        return data
        
    except json.JSONDecodeError as e:
        # Lanza una excepción para que el endpoint la capture
        raise ValueError(f"El modelo AI generó un JSON inválido. Salida cruda: {generated_text}")
    except Exception as e:
        # Lanza una excepción para cualquier otro error
        raise ValueError(f"Error al procesar la respuesta de la IA. Salida cruda: {generated_text}. Detalle del error: {str(e)}")
