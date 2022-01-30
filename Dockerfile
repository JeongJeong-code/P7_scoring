

FROM python:3.8-slim-buster
WORKDIR C:\Users\nwenz\Desktop\P7_scoring


COPY requirements.txt ./requirements.txt


RUN pip install -r requirements.txt
COPY df_train.csv ./df_train.csv
COPY df_test.csv ./df_test.csv

COPY train.py ./train.py
COPY inference.py ./inference.py
RUN python3 train.py