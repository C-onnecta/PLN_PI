from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import unicodedata

nltk.download('stopwords')

df_treinamento = pd.read_excel('Base_Pedidos.xlsx', engine='openpyxl')
df_treinamento['request_normalized'] = df_treinamento['request']

acentos_para_sem_acento = {
    'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a', 'ä': 'a',
    'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
    'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
    'ó': 'o', 'ò': 'o', 'õ': 'o', 'ô': 'o', 'ö': 'o',
    'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
    'ç': 'c',
    'Á': 'A', 'À': 'A', 'Ã': 'A', 'Â': 'A', 'Ä': 'A',
    'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
    'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
    'Ó': 'O', 'Ò': 'O', 'Õ': 'O', 'Ô': 'O', 'Ö': 'O',
    'Ú': 'U', 'Ù': 'U', 'Û': 'U', 'Ü': 'U',
    'Ç': 'C'
}

def remover_acentos(texto):
    texto_sem_acento = ""
    for char in texto:
        if char in acentos_para_sem_acento:
            texto_sem_acento += acentos_para_sem_acento[char]
        else:
            texto_sem_acento += char
    return texto_sem_acento

df_treinamento['request_normalized'] = df_treinamento['request_normalized'].apply(
    lambda x: re.sub(r'[^a-z0-9\s]', '', remover_acentos(x).lower())
)

df_treinamento['request_normalized'] = df_treinamento['request_normalized'].apply(lambda x: x.split())

stop_words_manual = {
    'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'e',
    'com', 'nao', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as',
    'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'a', 'seu',
    'sua', 'ou', 'ser', 'quando', 'muito', 'ha', 'nos', 'ja', 'esta',
    'eu', 'também', 'so', 'pelo', 'pela', 'ate', 'isso', 'ela', 'entre'
}

def remove_stopwords(words):
    return [word for word in words if word not in stop_words_manual]

df_treinamento['request_normalized'] = df_treinamento['request_normalized'].apply(remove_stopwords)

df_treinamento['request_normalized_str'] = df_treinamento['request_normalized'].apply(lambda x: ' '.join(x))
vectorizer = CountVectorizer(max_features=1000)

vector = vectorizer.fit_transform(df_treinamento['request_normalized_str'])
X = pd.DataFrame.sparse.from_spmatrix(vector, columns=vectorizer.get_feature_names_out())

y = df_treinamento['truthfulness']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dimensões do conjunto de treino:", X_train.shape)
print("Dimensões do conjunto de teste:", X_test.shape)
print("Rótulos de treino:", y_train.shape)
print("Rótulos de teste:", y_test.shape)

from sklearn.linear_model import LogisticRegression
modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score

y_pred = modelo.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

def remover_acentos(texto):
    nfkd = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def preprocessar_texto(texto):
    texto = remover_acentos(texto.lower())
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    palavras = texto.split()
    palavras = [word for word in palavras if word not in stop_words_manual]
    return ' '.join(palavras)

def classificar_frase(frase):
    frase_preprocessada = preprocessar_texto(frase)
    frase_vectorizada = vectorizer.transform([frase_preprocessada])
    previsao = modelo.predict(frase_vectorizada)

    if previsao[0] == 0:
        return "0 - Pedido considerado falso"
    else:
        return "1 - O pedido parece verídico"


# FastAPI
app = FastAPI()

class RequestInput(BaseModel):
    request: str

@app.post("/classify")
async def classify(input: RequestInput):
    return {"classification": classificar_frase(input.request)[4:]}
    
#Rodar o servidor: uvicorn RequestClassifier:app