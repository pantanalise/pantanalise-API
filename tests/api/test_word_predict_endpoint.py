import os

from fastapi.testclient import TestClient
from pantanalise.main import app


def test_word_predict_endpoint(mocker):
    client = TestClient(app)
    data = 'louro quer biscoito'
    predict_repository =  {'data': ['bolacha']}
    mock_word_predict_bentoml_repository = mocker.patch(
        'pantanalise.repository.bentoml.word_predict_bentoml_repository.WordPredictBentoMLRepository.predict'
    )
    mock_word_predict_bentoml_repository.return_value = predict_repository
    mocker.patch.dict(os.environ, {'BENTO_ML_INTEGRATION': 'true'})
    response = client.post("predict/token", json={'text': data})
    mock_word_predict_bentoml_repository.assert_called_once_with(data)
    assert response.status_code == 200
    assert response.json() == {'recommendWord': {'data': ['bolacha']}}
