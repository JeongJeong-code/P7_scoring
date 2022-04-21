FROM python:3.8
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY ./start.sh /start.sh
COPY ./app /app
RUN chmod +x start.sh

CMD ["./start.sh"]