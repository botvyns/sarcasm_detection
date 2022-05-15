# Модель розпізнавання сарказму на основі постів в соц. мережі твіттер та з використанням претренованих ваг Glove

### Для створення проекту використано

- python3.10

- tensorflow

- snscrape

- numpy

- [GloVe pretrained waights](https://lang.org.ua/en/models/#anchor4)

### Структура проекту

train_model.py - тренування моделі, результат буде збережено в папку checkpoints

load_model.py - використання натренованої моделі для визначення сарказму

dataset.json - неочищені дані

cleared_dataset.json - очищені дані

stopwords.json - стоп слова, оригінал взято з [GitHub](https://github.com/skupriienko/Ukrainian-Stopwords)

### Для початку роботи необхідно завантажити та додати в директорію файл Glove

fiction.cased.tokenized.glove.300d - https://lang.org.ua/static/downloads/models/fiction.cased.tokenized.glove.300d.bz2
