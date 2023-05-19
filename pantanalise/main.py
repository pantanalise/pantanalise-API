import distutils
import os
from os.path import join, dirname

from dotenv import load_dotenv
from fastapi import FastAPI
from os import environ

from uvicorn import run

from pantanalise.repository.bentoml.engage_predict_bentoml_repository import EngagePredictBentoMLRepository
from pantanalise.repository.bentoml.word_predict_bentoml_repository import WordPredictBentoMLRepository
from pantanalise.repository.model.engage_predict_direct_repository import predict
from pantanalise.repository.model.word_predict_direct_repository import WordPredictDirectRepository
from pantanalise.repository.model.sentiment_analysis_predict_direct_repository import SentimentAnalysisDirectRepository
from pantanalise.model_request.message_model import MessageModel

app = FastAPI()


@app.get("/health-check")
def read_root():
    return {"message": "Deu bom"}


@app.post("/predict/sentiment")
def predict_sentiment(body: MessageModel):
    sentiment_analysis_repository = SentimentAnalysisDirectRepository()
    sentiment_predict = sentiment_analysis_repository.predict(body.text)

    return {'sentimentWord' : sentiment_predict}

@app.post("/predict/token")
def predict_token(body: MessageModel):
    bentoml_integration_activate =   distutils.util.strtobool(environ.get("BENTO_ML_INTEGRATION"))
    if bentoml_integration_activate:
        word_predict_repository = WordPredictBentoMLRepository()
        word_recommend = word_predict_repository.predict(body.text)
    else:
        word_recommend = WordPredictDirectRepository().predict(body.text)

    return { "recommendWord" : word_recommend }

@app.post("/predict/engage")
def predict_engage(body: MessageModel):
    print(body.text)
    bentoml_integration_activate =   distutils.util.strtobool(environ.get("BENTO_ML_INTEGRATION"))
    if bentoml_integration_activate:
        pass
    else:
        word_recommend = predict(body.text)

    return { "recommendWord" : 'word_recommend' }

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