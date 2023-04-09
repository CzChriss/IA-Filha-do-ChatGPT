import openai
import requests
from bs4 import BeautifulSoup
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
import numpy as np
import requests
from bs4 import BeautifulSoup

def coletar_dados_web(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    dados = ""
    for p in soup.find_all('p'):
        dados += p.text + "\n"
    return dados

# Define as permissões necessárias para acessar a web e ler arquivos locais
os.environ["OPENAI_API_SECRET_KEY"] = "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"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

openai.api_key = "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"

# Função para coletar dados relevantes do domínio da IA
def coletar_dados():
    while True:
        resposta = input("Deseja coletar dados manualmente (m), com o ChatGPT (c) ou da web (w)? ")
        if resposta.lower() == "m":
            dados = coletar_dados_manual()
            break
        elif resposta.lower() == "c":
            n_samples = int(input("Quantos pares pergunta-resposta deseja coletar? "))
            dados = coletar_dados_chatgpt(n_samples)
            break
        elif resposta.lower() == "w":
            url = input("Digite a URL da página que deseja coletar dados: ")
            dados = coletar_dados_web(url)
            break
        else:
            print("Opção inválida. Digite 'm' para coletar dados manualmente, 'c' para coletar dados com o ChatGPT ou 'w' para coletar dados da web.")

# Função para coletar dados relevantes do domínio da IA
def coletar_dados_manual():
    dados = ""
    while True:
        # Obtém a próxima pergunta para a IA responder
        resposta = input("Digite sua pergunta: ")

        # Usa a API da OpenAI para obter a resposta
        response = openai.Completion.create(
            engine="davinci",
            prompt=resposta,
            max_tokens=2039,
            n=1,
            stop=None,
            temperature=0.7,
        )
        answer = response.choices[0].text.strip()

        # Adiciona a pergunta e resposta aos dados coletados
        dados += "Pergunta: " + resposta + "\n"
        dados += "Resposta: " + answer + "\n\n"

        # Pergunta ao usuário se deseja fazer outra pergunta
        resposta = input("Deseja fazer outra pergunta? (s/n) ")
        if resposta.lower() == "n":
            break
    
    return dados

    # Pré-processamento e limpeza dos dados
    dados = re.sub(r'\n+', ' ', dados)  # remover quebras de linha
    dados = re.sub(r'\[[0-9]*\]', '', dados)  # remover referências numéricas
    dados = re.sub(r'[^\w\s]', '', dados)  # remover pontuação
    dados = dados.lower()  # converter para minúsculas
    tokens = word_tokenize(dados)  # tokenizar palavras
    stop_words = set(stopwords.words('portuguese'))  # definir as palavras de parada em português
    tokens = [w for w in tokens if not w in stop_words]  # remover as palavras de parada
    dados = ' '.join(tokens)  # juntar as palavras tokenizadas em uma única string
    
    return dados
def coletar_dados_chatgpt(n_samples):
    dados = ""
    for i in range(n_samples):
        # Obtém a próxima pergunta para a IA responder
        resposta = input("Digite sua pergunta: ")

        # Usa a API da OpenAI para obter a resposta
        response = openai.Completion.create(
            engine="davinci",
            prompt=resposta,
            max_tokens=2039,
            n=1,
            stop=None,
            temperature=0.7,
        )
        answer = response.choices[0].text.strip()

        # Adiciona a pergunta e resposta aos dados coletados
        dados += "Pergunta: " + resposta + "\n"
        dados += "Resposta: " + answer + "\n\n"
    
    return dados

# Função para treinar a IA
def treinar_IA(dados):
    # Dividir os dados em conjunto de treinamento e teste
    dados = dados.split('\n')
    X = []
    y = []
    for linha in dados:
        if linha.startswith('Pergunta:'):
            X.append(linha[10:])
        elif linha.startswith('Resposta:'):
            y.append(linha[10:])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Tokenizar as perguntas e respostas
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train + y_train)
    vocab_size = len(tokenizer.word_index) + 1

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    y_train_seq = tokenizer.texts_to_sequences(y_train)
    y_test_seq = tokenizer.texts_to_sequences(y_test)

    # Padronizar o comprimento das sequências de perguntas e respostas
    max_len = max([len(seq) for seq in X_train_seq])
    X_train_seq = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_seq = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

    max_len = max([len(seq) for seq in y_train_seq])
    y_train_seq = pad_sequences(y_train_seq, maxlen=max_len, padding='post')
    y_test_seq = pad_sequences(y_test_seq, maxlen=max_len, padding='post')

    # Definir e treinar o modelo
    embedding_dim = 50
    lstm_units = 64

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
    model.add(LSTM(units=lstm_units))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train_seq, y_train_seq, epochs=10, validation_data=(X_test_seq, y_test_seq))

    # Avaliar a precisão e coerência do modelo
    loss, accuracy = model.evaluate(X_test_seq, y_test_seq)
    print('Precisão do modelo: {:.2f}%'.format(accuracy*100))

    return model

# Função para gerar a resposta da IA
def gerar_resposta(pergunta, model):
    pergunta_seq = tokenizer.texts_to_sequences([pergunta])
    pergunta_seq = pad_sequences(pergunta_seq, maxlen=max_len, padding='post')

    resposta_seq = np.argmax(model.predict(pergunta_seq), axis=-1)
    resposta = tokenizer.sequences_to_texts(resposta_seq)

    return resposta

# Função principal do script
def main():
    while True:
        resposta = input("Deseja treinar o modelo com dados coletados manualmente (m) ou com o ChatGPT (c)? ")
        if resposta.lower() == "m":
            dados = coletar_dados_manual()
            break
        elif resposta.lower() == "w":
            n_pages = int(input("Quantas páginas da web deseja coletar? "))
            dados = coletar_dados_web(n_pages)
            break
        elif resposta.lower() == "c":
            n_samples = int(input("Quantos pares pergunta-resposta deseja coletar? "))
            dados = coletar_dados_chatgpt(n_samples)
            break
        else:
            print("Resposta inválida. Digite 'm' para coletar dados manualmente ou 'w' para coletar dados da web.")

    model = treinar_IA(dados)

    while True:
        pergunta = input("Digite sua pergunta ou digite 'sair' para encerrar: ")
        if pergunta.lower() == "sair":
            break
        resposta = gerar_resposta(pergunta, model)
        print(resposta)

if __name__ == "__main__":
    main()
