# VK CUP 2021
## ML - трек. Отборочный раунд.
---
[https://cups.mail.ru/ru/contests/vkcup2021](https://cups.mail.ru/ru/contests/vkcup2021)
---
Локальное обучение модели:
```
python -m pip install -r requirements.txt
python solve.py
```

Создание docker-контейнера и загрузка в репозиторий:
```
docker build -t <local tag> .
docker login stor.highloadcup.ru -u <login> -p <password>
docker tag vkcup stor.highloadcup.ru/vkcup21_age/<remote tag>
docker push stor.highloadcup.ru/vkcup21_age/<remote tag>
```
## Результат
* Local: 9.0718
* Public: 8.8499 
* Private: 8.9388

66 место из 115
