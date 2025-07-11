from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

app = FastAPI()

model = init_chat_model("command-r-plus", model_provider="cohere")

chat_history = [
    SystemMessage(content="Eres un asistente inteligente que ayuda a los clientes en la compra de galletas, las galletas disponibles son de chocolate, vainilla y fresa, tienen un costo de tres dolares. No respondas nada que no sea relacionado a la compra de galletas."),
]

class Bot(BaseModel):
    query: str

@app.post("/bot")
async def bot(q: Bot):
    question=q.query

    chat_history.append(HumanMessage(question))

    response = model.invoke(chat_history)

    chat_history.append(response)

    return chat_history[-1].content