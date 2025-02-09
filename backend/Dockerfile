# Python 3.9の公式イメージをベースとして使用
FROM python:3.9

# 作業ディレクトリを設定
WORKDIR /app

# pipをアップグレード
RUN pip install --upgrade pip

# 必要なパッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードをコピー
COPY . .

# workflows.yaml ファイルをコピー
COPY workflows.yaml .

# ポート42069を公開
EXPOSE 42069

# アプリケーションを起動
CMD ["python", "app.py"]
