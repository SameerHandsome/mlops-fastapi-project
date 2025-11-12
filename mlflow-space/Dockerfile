FROM python:3.10-slim
RUN pip install mlflow[extras]
EXPOSE 5000
COPY start_mlflow.sh /start_mlflow.sh
ENTRYPOINT ["bash", "/start_mlflow.sh"]