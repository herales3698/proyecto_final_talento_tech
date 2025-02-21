from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import nltk
import statistics
from typing import Dict, Tuple

# ----------------------------
# Configuración y carga de datos
# ----------------------------

# Configurar NLTK: se añade la ruta y se descargan los recursos necesarios
nltk.data.path.append('C:\\Users\\hemor\\AppData\\Roaming\\nltk_data')
nltk.download('punkt')
nltk.download('wordnet')

def load_data():
    """
    Carga y prepara el dataset para el análisis.
    Se seleccionan las columnas relevantes y se normalizan los nombres.
    """
    data = pd.read_csv('Dataset/dataset_consumo_energia_balanceado.csv')[['Año', 'Mes', 'Personas', 'Consumo']]
    data.columns = ['Year', 'Month', 'People', 'Consumption']  # Normalizar nombres de columnas
    return data.fillna('').to_dict(orient='records')  # Devolver lista de diccionarios

# Cargar datos de consumo energético
energy_data = load_data()

def get_consumption_level(consumption_per_person: float):
    """
    Determina el nivel de consumo (alto, medio o bajo) basado en estadísticas históricas.
    Calcula la media y la desviación estándar de consumo por persona.
    """
    consumptions_per_person = [record['Consumption'] / record['People'] 
                               for record in energy_data if record['People'] > 0]
    
    avg_consumption = statistics.mean(consumptions_per_person)
    std_consumption = statistics.stdev(consumptions_per_person)
    
    if consumption_per_person > (avg_consumption + std_consumption):
        level = "high"
    elif consumption_per_person < (avg_consumption - std_consumption):
        level = "low"
    else:
        level = "medium"
    
    return level, avg_consumption, std_consumption

def get_recommendations(consumption: float, people: int) -> Dict[str, list]:
    """
    Genera recomendaciones personalizadas basadas en el consumo y el número de personas.
    Retorna tanto recomendaciones generales como específicas según el nivel de consumo.
    """
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
            f"El consumo por persona es {consumption_per_person:.2f} kWh, intenta reducirlo",
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

def get_additional_recommendations(details: str) -> list:
    """
    Genera recomendaciones adicionales basadas en los detalles extra proporcionados por el usuario.
    Se detectan palabras clave para personalizar la respuesta.
    """
    extra_recommendations = [
        "Revisa el mantenimiento de tus equipos de calefacción y refrigeración",
        "Considera utilizar reguladores de voltaje para optimizar el consumo",
        "Implementa un sistema de monitoreo para identificar picos de consumo",
        "Asegúrate de que las ventanas estén bien aisladas",
        "Evalúa la posibilidad de instalar sistemas de energía renovable, como paneles solares"
    ]
    
    # Personalización simple según palabras clave en los detalles
    if "calefacción" in details.lower():
        extra_recommendations.append("Optimiza el uso de la calefacción, ajusta el termostato y revisa el sistema regularmente.")
    if "iluminación" in details.lower():
        extra_recommendations.append("Considera instalar sensores de movimiento y temporizadores para la iluminación.")
    if "electrodomésticos" in details.lower():
        extra_recommendations.append("Revisa la eficiencia energética de tus electrodomésticos y reemplázalos por modelos más eficientes si es posible.")
    if "computador" in details.lower():
        extra_recommendations.append("Revisa la eficiencia energética de tus equipos de computación y reemplázalos por modelos más eficientes si es posible.")
    if "aire acondicionado" in details.lower():
        extra_recommendations.append("Optimiza el uso del aire acondicionado, ajusta el termostato y revisa el sistema regularmente.")
    if "luces" in details.lower():
        extra_recommendations.append("Revisa la eficiencia energética de tus luces y reemplázalas por modelos más eficientes si es posible.")
    return extra_recommendations

# ----------------------------
# Lógica Conversacional del Chatbot
# ----------------------------

def process_message(state: dict, message: str) -> Tuple[dict, str]:
    """
    Procesa el mensaje del usuario según el estado actual de la conversación.
    Actualiza el estado y retorna la respuesta correspondiente.
    
    Estados definidos:
      - step 0: Espera el nombre.
      - step 1: Espera el número de personas en el hogar.
      - step 2: Espera el consumo mensual en kWh.
      - step 3: Muestra recomendaciones y pregunta si se desean más detalles.
      - step 4: Procesa detalles adicionales y ofrece recomendaciones extra.
    """
    step = state.get("step", 0)
    reply = ""
    
    if step == 0:
        # Guardar el nombre y avanzar al siguiente paso
        state["name"] = message.strip()
        state["step"] = 1
        reply = f"Encantado, {state['name']}. ¿Cuántas personas hay en tu hogar?"
    
    elif step == 1:
        # Validar e interpretar el número de personas
        try:
            people = int(message)
            if people <= 0:
                reply = "Por favor, ingresa un número válido de personas (mayor que 0)."
            else:
                state["people"] = people
                state["step"] = 2
                reply = "Perfecto. ¿Cuál es el consumo mensual en kWh?"
        except ValueError:
            reply = "Por favor, ingresa un número válido para el número de personas."
    
    elif step == 2:
        # Validar e interpretar el consumo
        try:
            consumption = float(message)
            if consumption <= 0:
                reply = "El consumo debe ser mayor que 0. Inténtalo de nuevo."
            else:
                state["consumption"] = consumption
                state["step"] = 3
                # Generar recomendaciones basadas en los datos
                recs = get_recommendations(consumption, state["people"])
                # Construir mensaje con resumen y recomendaciones
                reply_lines = [
                    f"Análisis realizado para {state['name']}:",
                    f"Consumo mensual: {consumption} kWh con {state['people']} personas.",
                    f"Nivel de consumo: {recs['consumption_level'].capitalize()}",
                    "Recomendaciones generales:"
                ]
                reply_lines.extend([f"- {rec}" for rec in recs["general"]])
                reply_lines.append("Recomendaciones específicas:")
                reply_lines.extend([f"- {rec}" for rec in recs["specific"]])
                reply_lines.append("¿Deseas proporcionar más detalles para obtener recomendaciones adicionales? (responde 'sí' o 'no')")
                reply = "\n".join(reply_lines)
        except ValueError:
            reply = "Por favor, ingresa un número válido para el consumo."
    
    elif step == 3:
        # Procesar respuesta para detalles adicionales
        if message.strip().lower() in ["sí", "si"]:
            state["step"] = 4
            reply = "Por favor, proporciona más detalles sobre tu situación (ej. problemas específicos o áreas a mejorar):"
        else:
            reply = "¡Gracias por utilizar nuestro servicio, que tengas un buen día!"
            # Reiniciar el estado para comenzar una nueva conversación
            state = {"step": 0}
    
    elif step == 4:
        # Procesar los detalles adicionales y generar recomendaciones extra
        state["details"] = message.strip()
        extra_recs = get_additional_recommendations(state["details"])
        reply_lines = ["Basado en los detalles que proporcionaste, aquí tienes algunas recomendaciones adicionales:"]
        reply_lines.extend([f"- {rec}" for rec in extra_recs])
        reply = "\n".join(reply_lines)
        # Reiniciar el estado tras finalizar la conversación
        state = {"step": 0}
    
    else:
        reply = "Lo siento, ha ocurrido un error en la conversación."
    
    return state, reply

# ----------------------------
# Endpoints de FastAPI
# ----------------------------

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def chat_page():
    """
    Devuelve la página principal con la interfaz del chatbot.
    Se incluye HTML, CSS y JavaScript para simular un chat en tiempo real.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chatbot de Consumo Energético</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f2f2f2; }
            #chat-container { width: 60%; margin: auto; margin-top: 50px; background-color: #fff; padding: 20px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            #chat-log { height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; white-space: pre-wrap; }
            .message { margin: 10px 0; }
            .user { color: blue; }
            .bot { color: green; }
            #input-form { margin-top: 20px; }
        </style>
    </head>
    <body>
        <div id="chat-container">
            <h1>Chatbot de Consumo Energético</h1>
            <div id="chat-log"></div>
            <form id="input-form">
                <input type="text" id="user-input" placeholder="Escribe tu mensaje aquí" style="width:80%;" autocomplete="off" required/>
                <button type="submit">Enviar</button>
            </form>
        </div>
        
        <script>
            // Estado inicial de la conversación
            let state = { "step": 0 };
            
            // Función para añadir mensajes al registro del chat
            function appendMessage(sender, message) {
                const chatLog = document.getElementById("chat-log");
                const messageDiv = document.createElement("div");
                messageDiv.classList.add("message");
                messageDiv.classList.add(sender);
                messageDiv.textContent = sender.toUpperCase() + ": " + message;
                chatLog.appendChild(messageDiv);
                chatLog.scrollTop = chatLog.scrollHeight;
            }
            
            // Mensaje inicial del bot
            window.onload = function() {
                appendMessage("PowerBot", "¡Hola! ¿Cuál es tu nombre?");
            };
            
            // Manejo del envío del formulario
            document.getElementById("input-form").addEventListener("submit", function(e) {
                e.preventDefault();
                const userInput = document.getElementById("user-input");
                const message = userInput.value;
                appendMessage("user", message);
                userInput.value = "";
                
                // Enviar mensaje y estado al endpoint /chat
                fetch("/chat", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ "state": state, "message": message })
                })
                .then(response => response.json())
                .then(data => {
                    state = data.state;
                    appendMessage("PowerBot", data.reply);
                })
                .catch(error => {
                    console.error("Error:", error);
                });
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/chat")
async def chat_endpoint(payload: dict):
    """
    Endpoint que recibe el mensaje del usuario y el estado actual de la conversación.
    Retorna la respuesta generada y el estado actualizado en formato JSON.
    """
    state = payload.get("state", {"step": 0})
    message = payload.get("message", "")
    new_state, reply = process_message(state, message)
    return JSONResponse(content={"state": new_state, "reply": reply})
