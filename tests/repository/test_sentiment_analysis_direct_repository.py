from pantanalise.repository.model.sentiment_analysis_direct_repository import SentimentAnalysisDirectRepository


def test_sentiment_analysis_direct_repository():
    tweet = "Além disso, considerou o Brasil um lider na promoção do esporte escolar no mundo após a volta dos Jogos Escolares Brasileiros (JEBs), passados mais de 15 anos sem sua realização"
    tweet= 'eu odeio e amo todo mundo nessa bosta '
    result  = SentimentAnalysisDirectRepository().predict(tweet)
    print(result)