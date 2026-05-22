FROM python:3.8-alpine3.15

WORKDIR /code

RUN pip install --no-cache-dir --upgrade fastapi uvicorn loguru

COPY . /code

ENV PYTHONPATH "${PYTHONPATH}:/code/"


EXPOSE 8050

CMD ["uvicorn", "auth_server:app", "--host", "0.0.0.0", "--port", "8050"]
