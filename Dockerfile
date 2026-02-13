FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential

# 修正: バージョンを指定してインストール
RUN pip install --no-cache-dir \
    torch \
    transformers==4.35.2 \
    accelerate

COPY . /app

CMD ["python3"]