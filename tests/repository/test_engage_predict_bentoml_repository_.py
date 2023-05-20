from pantanalise.repository.bentoml.engage_predict_bentoml_repository import EngagePredictBentoMLRepository


def test_predict_with_bentoml(mocker):
    url_model = 'http://localhost:32455'
    request_mock = mocker.patch('requests.post')
    mocker.patch("os.environ.get", return_value=url_model)
    data = 'louro quer biscoito'
    request_mock.return_value.json.return_value = {'data':  [423,252]}
    repository = EngagePredictBentoMLRepository()
    response = repository.predict(data)
    assert response == {'data':  [423,252]}
    request_mock.assert_called_once_with(url_model, data=data)
