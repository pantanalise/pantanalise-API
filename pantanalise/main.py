from functools import reduce
from os.path import join, dirname

from dotenv import load_dotenv
from fastapi import FastAPI
from os import environ

from uvicorn import run

from pantanalise.model.sentiment_analysis_repository import SentimentAnalysisRepository
from pantanalise.model_request.message_model import MessageModel

app = FastAPI()


@app.get("/health-check")
def read_root():
    return {"message": "Deu bom"}


@app.post("/predict")
def predict(body: MessageModel):
    sentiment_analysis_repository = SentimentAnalysisRepository()
    sentiment_predict = sentiment_analysis_repository.predict(body.text)

    return { "feeling" : sentiment_predict }



def start_server():
    dotenv_path = join(dirname(__file__), "../.env")
    load_dotenv(dotenv_path)
    run(
        "pantanalise.main:app",
        host=environ.get("HOST"),
        port=int(environ.get("PORT")),
        reload=True,
    )


if __name__ == "__main__":
    start_server()