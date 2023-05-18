from pantanalise.repository.bentoml.word_predict_bentoml_repository import WordPredictBentoMLRepository


def test_predict_with_bentoml(mocker):
    url_model = 'http://localhost:32452'
    request_mock = mocker.patch('requests.post')
    mocker.patch("os.environ.get", return_value=url_model)
    data = 'louro quer biscoito'
    request_mock.return_value.json.return_value = {'data': ['bolacha']}
    repository = WordPredictBentoMLRepository()
    response = repository.predict(data)
    assert response == {'data': ['bolacha']}
    request_mock.assert_called_once_with(url_model, data=data)
