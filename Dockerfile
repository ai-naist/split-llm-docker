FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential

# 最新のtransformersとaccelerate、Qwenに必要なライブラリを入れる
RUN pip install --no-cache-dir \
    torch \
    transformers \
    accelerate \
    tiktoken \
    blobfile

COPY . /app

CMD ["python3"]