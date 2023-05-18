import time
from os.path import join, dirname

from transformers import AutoTokenizer

from pantanalise.repository.predict import Predict
from torch import load

masked_lang_model = load(join(dirname(__file__), "final_masked_language_model.pth"))

from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

mask_filler = pipeline(
    "fill-mask", model=masked_lang_model, tokenizer=tokenizer, device="cuda:0"
)

class WordPredictDirectRepository(Predict):

    def predict(self, data):
        inicio = time.time()
        tweet = data
        palavras = tweet.split()
        data_predict = []
        data = {}
        # Imprime as palavras usando um loop
        for i in range(len(palavras)):
            print('predição palavra ')
            palavras_mascaradas = palavras.copy()
            palavras_mascaradas[i] = "[MASK]"
            tweet =  ' '.join(palavras_mascaradas)
            preds = mask_filler(tweet)

            for pred in preds:
                data_predict.append(pred['sequence'])
                print(f">>> {pred['sequence']}")
            data.update({palavras[i]: data_predict })
        fim = time.time()
        tempo_execucao = fim - inicio
        print("Tempo de execução: {:.2f} segundos".format(tempo_execucao))
        return data

