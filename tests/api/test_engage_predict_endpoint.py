import os

from fastapi.testclient import TestClient
from pantanalise.main import app


def test_engage_predict_endpoint(mocker):
    client = TestClient(app)
    data = 'louro quer biscoito'
    predict_repository =  {'data': [423,232]}
    mock_word_predict_bentoml_repository = mocker.patch(
        'pantanalise.repository.bentoml.engage_predict_bentoml_repository.EngagePredictBentoMLRepository.predict'
    )
    mock_word_predict_bentoml_repository.return_value = predict_repository
    mocker.patch.dict(os.environ, {'BENTO_ML_INTEGRATION': 'true'})
    response = client.post("predict/engage", json={'text': data})
    mock_word_predict_bentoml_repository.assert_called_once_with(data)
    assert response.status_code == 200
    assert response.json() == {'like': 423, 'retweets': 232}
