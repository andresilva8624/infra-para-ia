import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="API de Sentimento — Infraestrutura Computacional para IA",
    description="Aula 1: nossa primeira aplicação de IA dentro de um container.",
    version="1.0",
)

modelo = joblib.load("modelo.pkl")


class Entrada(BaseModel):
    texto: str


class Saida(BaseModel):
    sentimento: str
    confianca: float


class Sobre(BaseModel):
    nomes: str


@app.get("/")
def raiz():
    """Verificação de saúde: útil para saber se o container está no ar."""
    return {
        "status": "ok",
        "servico": "api-de-sentimento",
        "aula": 1
    }


@app.post("/prediz", response_model=Saida)
def prediz(entrada: Entrada):
    """Classifica o sentimento de uma frase em português."""
    probabilidades = modelo.predict_proba([entrada.texto])[0]
    indice = probabilidades.argmax()

    return Saida(
        sentimento=modelo.classes_[indice],
        confianca=round(float(probabilidades[indice]), 4),
    )


@app.post("/sobre", response_model=Sobre)
def sobre():
    return Sobre(
        nomes="Andre Silva & Rafaella Santos"
    )