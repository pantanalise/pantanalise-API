import os

import requests

from pantanalise.repository.predict import Predict


class EngagePredictBentoMLRepository(Predict):

    def predict(self, data):
        url = str(os.environ.get("MODEL_PREDICT_ENGAGE_URL"))
        response = requests.post(url, data=data)
        return response.json()
