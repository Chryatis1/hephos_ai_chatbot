from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Hephos AI Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "Hephos AI Chatbot is running!"}

@app.post("/chat")
def chat(req: ChatRequest):
    system_prompt = (
        "You are Hephos AI Assistant. Be concise, helpful, and professional. "
        "If the user writes Croatian, reply in Croatian; otherwise reply in English."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":req.message},
            ],
            max_tokens=350, temperature=0.7,
        )
        return {"response": resp.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}
