# ビルドステージ
FROM node:14 AS build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# 実行ステージ
FROM nginx:alpine

# ビルドしたファイルをNginxのドキュメントルートにコピー
COPY --from=build /app/build /usr/share/nginx/html

# Nginxのデフォルト設定を上書き
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
