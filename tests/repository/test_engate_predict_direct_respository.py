from pantanalise.repository.model.engage_predict_direct_repository import EngagePredictDirectRepository


def test_engage_predict_repository():
    tweet = "Além disso, considerou o Brasil um lider na promoção do esporte escolar no mundo após a volta dos Jogos Escolares Brasileiros (JEBs), passados mais de 15 anos sem sua realização"
    tweet= 'eu odeio e amo todo mundo nessa bosta '
    result  = EngagePredictDirectRepository().predict(tweet)
    print(result)