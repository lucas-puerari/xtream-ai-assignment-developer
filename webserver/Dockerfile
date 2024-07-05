FROM python:3.12.4-slim

WORKDIR /webserver

COPY ./requirements.txt /webserver/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /webserver/requirements.txt

COPY . /webserver

EXPOSE 80

CMD ["fastapi", "run", "src/main.py", "--port", "8000"]