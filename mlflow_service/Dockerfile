FROM docker.arvancloud.ir/python:3.10-slim

RUN pip install mlflow==2.13.2

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "/mlruns"]