from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import nltk.corpus
from typing import Dict
import statistics

nltk.data.path.append('C:\\Users\\hemor\\AppData\\Roaming\\nltk_data')

nltk.download('punkt')
nltk.download('wordnet')

def load_data():
    data = pd.read_csv('Dataset/dataset_consumo_energia_balanceado.csv')[['Año', 'Mes', 'Personas', 'Consumo']]
    data.columns = ['Year', 'Month', 'People', 'Consumption']
    return data.fillna('').to_dict(orient='records')

energy_data = load_data()

def get_consumption_level(consumption_per_person: float):
    """Determina si el consumo por persona es alto, medio o bajo basado en datos históricos"""

    # Obtener datos históricos normalizados (consumo por persona)
    consumptions_per_person = [record['Consumption'] / record['People'] for record in energy_data if record['People'] > 0]

    # Calcular estadísticas
    avg_consumption = statistics.mean(consumptions_per_person)
    std_consumption = statistics.stdev(consumptions_per_person)

    # Determinar nivel de consumo
    if consumption_per_person > (avg_consumption + std_consumption):
        level = "high"
    elif consumption_per_person < (avg_consumption - std_consumption):
        level = "low"
    else:
        level = "medium"

    # Retornamos el nivel junto con la media y desviación estándar
    return level, avg_consumption, std_consumption


def get_recommendations(consumption: float, people: int) -> Dict[str, list]:
    """Genera recomendaciones personalizadas basadas en el nivel de consumo"""
    consumption_per_person = consumption / people
    consumption_level, avg_consumption, std_consumption = get_consumption_level(consumption_per_person)

    general_recommendations = [
        "Apaga las luces cuando no estés en la habitación",
        "Utiliza bombillas LED de bajo consumo",
        "Desconecta los aparatos electrónicos cuando no los uses",
        "Aprovecha la luz natural durante el día"
    ]

    specific_recommendations = {
        "high": [
            "Tu consumo es alto. Considera estas acciones adicionales:",
            "Revisa el aislamiento de tu hogar",
            "Programa el termostato a temperaturas más eficientes",
            f"El consumo por persona es {consumption_per_person:.2f} kWh, considera reducirlo",
            "Realiza una auditoría energética de tu hogar"
        ],
        "medium": [
            "Tu consumo es moderado. Puedes mejorarlo con estas acciones:",
            "Usa electrodomésticos en horas valle",
            "Instala temporizadores en equipos de alto consumo",
            f"El consumo por persona es {consumption_per_person:.2f} kWh"
        ],
        "low": [
            "¡Excelente! Tu consumo es bajo. Para mantenerlo:",
            "Continúa con tus buenos hábitos de consumo",
            "Considera instalar paneles solares para ser aún más eficiente",
            f"El consumo por persona es {consumption_per_person:.2f} kWh"
        ]
    }

    return {
        "general": general_recommendations,
        "specific": specific_recommendations[consumption_level],
        "consumption_level": consumption_level,
        "statistics": {
            "avg_consumption_per_person": f"{avg_consumption:.2f} kWh",
            "std_consumption_per_person": f"{std_consumption:.2f} kWh"
        }
    }

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <body>
            <h1>Chatbot de Consumo Energético</h1>
            <form action="/get_advice" method="get">
                <label>Consumo mensual (kWh):</label><br>
                <input type="number" name="consumption" step="0.01" required><br>
                <label>Número de personas:</label><br>
                <input type="number" name="people" required><br><br>
                <input type="submit" value="Obtener recomendaciones">
            </form>
        </body>
    </html>
    """

@app.get("/get_advice", tags=["Advice"])
async def get_advice(consumption: float, people: int):
    if consumption <= 0 or people <= 0:
        raise HTTPException(status_code=400, detail="Los valores deben ser mayores que 0")
    
    recommendations = get_recommendations(consumption, people)
    
    return JSONResponse(content={
        "message": f"Análisis de consumo para {people} personas con {consumption} kWh",
        "recommendations": recommendations
    })
