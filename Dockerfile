FROM python:3

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Для локальной отладки
COPY ./data/friends.csv /tmp/data/friends.csv
COPY ./data/train.csv /tmp/data/test.csv
COPY ./data/trainEducationFeatures.csv /tmp/data/testEducationFeatures.csv
COPY ./data/trainGroups.csv /tmp/data/testGroups.csv

COPY settings.py .
COPY utils.py .
COPY solve.py .
COPY model.pkl .
COPY friend_encoder.pkl .
COPY svd.pkl .

CMD python solve.py -m model.pkl
