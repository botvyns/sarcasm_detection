import json
import os
import re
import string

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix, classification_report

stopwords = json.load(open('stopwords.json', 'r', encoding='utf-8'))  # завантажуємо стоп слова
df = pd.read_json(open('dataset.json', 'r', encoding='utf-8'))  # завантаження датасету

"""
Очищуємо датасет від тегів та посилань
"""


def remove_sarcasm_word(text):
    for i in text.split():
        if i.lower() == 'сарказм' or i.lower() == 'сраказм':
            return text.replace(i, '')
    return text


df.text = df.text.apply(lambda x: re.sub(r'http\S+', '', x))
df.text = df.text.apply(lambda x: re.sub('@[^\s]+', '', x))
df.text = df.text.apply(lambda x: re.sub("#[а-яієїґА-ЯІЄЇҐA-Za-z0-9_]+", "", x))
df.text = df.text.apply(lambda x: x.replace("сарказм", "").replace('*', '')
                        .replace(')', '').replace('(', '').replace("тайтаке", "").replace("#", "").strip())
df.text = df.text.apply(lambda x: remove_sarcasm_word(x))

tokenizer_obj = Tokenizer()
max_length = 25


def tokenize_data(df):
    """
    Перетворюємо тексти в токени
    :param df: датасет
    :return: Токенізовані дані
    """
    tweets = list()
    lines = df["text"].values.tolist()
    for line in lines:
        # видаляємо спец символи
        line = re.sub(r'[^\w\s]', '', line).lower().split()
        words = [word for word in line if word not in stopwords]
        tweets.append(words)

    return tweets


def predict_sarcasm(string, model):
    """
    Виявляє сарказм та виводить результат на екран
    :param string: Текст
    :param model: модель tensorflow
    :return: Повертає рядок з ймовірністю сарказму
    """
    x_final = pd.DataFrame({"text": [string]})
    test_lines = tokenize_data(x_final)
    test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    test_review_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')
    pred = model.predict(test_review_pad)[0]
    print(pred)
    pred *= 100
    if pred >= 50:
        return f"{string} - It's a sarcasm!"
    else:
        return f"{string} - It's not a sarcasm."


"""
Завантаження даних в токенайзер tensorflow
"""
twitter_posts = tokenize_data(df)
validation_split = 0.2
tokenizer_obj.fit_on_texts(twitter_posts)
sequences = tokenizer_obj.texts_to_sequences(twitter_posts)
word_index = tokenizer_obj.word_index
vocab_size = len(tokenizer_obj.word_index) + 1

lines_pad = pad_sequences(sequences, maxlen=max_length, padding='post')
sentiment = df['is_sarcastic'].values

indices = np.arange(lines_pad.shape[0])
lines_pad = lines_pad[indices]
sentiment = sentiment[indices]
num_validation_samples = int(validation_split * lines_pad.shape[0])
"""
Поділ датасету на тестові та тренувальні дані
"""
X_train_pad = lines_pad[:-num_validation_samples]
y_train = sentiment[:-num_validation_samples]
X_test_pad = lines_pad[-num_validation_samples:]
y_test = sentiment[-num_validation_samples:]

"""
Завантажуємо glove ваги
"""
embeddings_index = {}
embedding_dim = 300
f = open(os.path.join('./fiction.cased.tokenized.glove.300d'), encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
c = 0
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        c += 1
        embedding_matrix[i] = embedding_vector


def create_model():
    """
    Створення моделі tensorflow
    :return: Повертає модель готову до тренування
    """
    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_length,
                                trainable=False)
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


if __name__ == "__main__":
    """
    Запускаємо код тільки якщо виконується поточний скрипт
    """
    from collections import Counter
    from wordcloud import WordCloud

    """
    Створюємо WordCloud для датасету
    """
    twitter_data = df.loc[df['is_sarcastic'] == 1]
    pos_twitter_lines = tokenize_data(twitter_data)
    pos_lines = [j for sub in pos_twitter_lines for j in sub]
    word_could_dict = Counter(pos_lines)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    wordcloud = WordCloud(width=850, height=600).generate_from_frequencies(word_could_dict)
    ax1.set_title('Сарказм')
    ax1.imshow(wordcloud)

    """
    Створюємо WordCloud для не саркастичних текстів
    """
    twitter_data = df.loc[df['is_sarcastic'] == 0]
    pos_twitter_lines = tokenize_data(twitter_data)
    pos_lines = [j for sub in pos_twitter_lines for j in sub]
    word_could_dict = Counter(pos_lines)

    wordcloud = WordCloud(width=850, height=600).generate_from_frequencies(word_could_dict)
    ax2.set_title('Не сарказм')
    ax2.imshow(wordcloud)
    plt.show()

    """
    Відображаємо кількість словоформ
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    text_len = df[df['is_sarcastic'] == 1]['text'].str.split().map(lambda x: len(x))
    ax1.hist(text_len, color='#7c00fe')
    ax1.set_title('Сарказм')
    text_len = df[df['is_sarcastic'] == 0]['text'].str.split().map(lambda x: len(x))
    ax2.hist(text_len, color='#f54200')
    ax2.set_title('Не сарказм')
    fig.suptitle('Кількість словоформ')
    plt.show()
    """
    Створюємо модель та тренуємо її
    """
    model = create_model()

    history = model.fit(X_train_pad, y_train, batch_size=32, epochs=25, validation_data=(X_test_pad, y_test), verbose=2)

    """
    Виводимо результати тренування
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'g', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    """
    Виводи метрики моделі
    """
    test_prediction = [True if i[0] * 100 > 50 else False for i in model.predict(X_test_pad)]
    print(classification_report(y_test, test_prediction, target_names=['Not Sarcastic', 'Sarcastic']))

    """
    Виводимо матрицю невідповідності
    """
    cm = confusion_matrix(y_test, test_prediction)
    cm = pd.DataFrame(cm, index=['Не сарказм', 'Сарказм'], columns=['Не сарказм', 'Сарказм'])
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='',
                xticklabels=['Не сарказм', 'Сарказм'], yticklabels=['Не сарказм', 'Сарказм'])
    plt.show()

    """
    Зберігаємо модель
    """
    model.save_weights('./checkpoints/sarcasm_checkpoint')
