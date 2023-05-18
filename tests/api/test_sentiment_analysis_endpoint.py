
from fastapi.testclient import TestClient
from pantanalise.main import app

def test_predict_direct_sentiment_world(mocker):
    client = TestClient(app)
    data =  'odeio odeio odeio'
    mock_sentiment_analysis_repository = mocker.patch(
        'pantanalise.repository.model.sentiment_analysis_direct_repository.SentimentAnalysisDirectRepository.predict',
    )
    mock_sentiment_analysis_repository.return_value = 'negativo'
    # os.environ.get = mocker.patch("os.environ.get", return_value=secret)
    response = client.post("/predict/sentiment", json={'text': data})
    mock_sentiment_analysis_repository.assert_called_once_with(data)
    assert response.status_code == 200
    assert response.json() == {"feeling": "negativo"}
