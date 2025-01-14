FROM python:3.11-slim

RUN pip install pipenv gunicorn

WORKDIR /app

COPY 'Pipfile' 'Pipfile.lock' ./

RUN pipenv install --system --deploy

COPY 'predict.py' 'linear_model.pkl' ./

EXPOSE 9990

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9990", "predict:app" ]