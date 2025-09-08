FROM python:3.12

WORKDIR /app

RUN pip install --no-cache-dir fastapi fastapi[standard] joblib uvicorn scikit-learn

COPY . /app

EXPOSE 5876

CMD ["uvicorn", "web_server:app", "--host", "0.0.0.0", "--port", "5876"]
