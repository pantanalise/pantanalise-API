import os

import requests

from pantanalise.repository.predict import Predict


class WordPredictBentoMLRepository(Predict):

    def predict(self, data):
        url = str(os.environ.get("MODEL_PREDICT_WORD_URL"))
        response = requests.post(url, data=data)
        return response.json()

