# Getting Started with Create React App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)



# SimulateX

SimulateX は、複数の AI モデル（GPT-4, Claude, Gemini）を統合し、企業の意思決定支援、シミュレーション、データ解析を行うプラットフォームです。

## 🏗 **プロジェクト構成**

simulatex/
│── backend/          # バックエンド（FastAPI / Flask, Weaviate, MongoDB）
│── frontend/         # フロントエンド（React / Vue.js）
│── public/           # 静的リソース（HTML, CSS, JavaScript）
│── docker-compose.yml # Docker 設定
│── README.md         # このファイル

---

## **1️⃣ Backend (`backend/`)**
### **📌 概要**
バックエンドは `FastAPI` または `Flask` をベースにし、各 LLM（GPT-4, Claude, Gemini）との連携、データ管理、API通信、ログ処理を担当。

### **🔹 主要機能**
- **マルチ LLM 連携**（GPT-4, Claude, Gemini）
- **Weaviate データベース** との接続 (`weaviate_connection_pool.py`)
- **ワークフロー管理** (`workflow_manager.py`)
- **データ処理・整形** (`data_manager.py`)
- **API サーバー機能** (`app.py`)
- **エージェント型 AI システム** (`agents/`)

### **📌 使用技術**
- `Python`
- `FastAPI / Flask`
- `Weaviate`
- `LangChain`
- `MongoDB`

### **⚙️ 起動方法**
```bash
cd backend
pip install -r requirements.txt
python app.py

🚧 開発途中の機能
	•	✅ API認証: JWT認証の導入（開発中）
	•	✅ ログ管理: エラーログの詳細記録（調整中）
	•	✅ マルチ LLM の統合: 設定ファイルを使い AI の切り替えを可能にする（テスト中）
	•	🛠️ WebSocket: リアルタイム応答用に WebSocket 実装（計画中）

2️⃣ Public (public/)

📌 概要

このディレクトリにはフロントエンドの静的リソースが格納されており、React や Vue.js で作られた UI のベースを提供。

📌 主要ファイル
	•	index.html → フロントエンドのエントリーポイント
	•	favicon.ico → アプリアイコン
	•	logo192.png, logo512.png → ロゴ画像
	•	manifest.json → PWA 対応設定
	•	robots.txt → SEO 設定

📌 使用技術
	•	HTML
	•	CSS
	•	JavaScript
        •       Python       

🚧 開発途中の機能
	•	✅ ダークモード: ユーザー設定に応じた UI カスタマイズ（デザイン調整中）
	•	✅ モバイル最適化: スマホ対応（UI 調整中）

3️⃣ Frontend (src/)

📌 概要

フロントエンドは React と Vue.js を採用し、ダッシュボード・データ表示・AI モデルとのやり取りを担当。

📌 主要ファイル
	•	App.js → メインのコンポーネント
	•	index.js → React アプリのエントリーポイント
	•	components/ → 各種 UI コンポーネント
	•	reportWebVitals.js → パフォーマンス計測
	•	setupTests.js → ユニットテスト設定

📌 使用技術
	•	React
	•	Vue.js
	•	JavaScript (ES6+)
	•	CSS / SCSS

⚙️ 起動方法

cd frontend
npm install
npm start

🚧 開発途中の機能
	•	✅ UI/UX 改善: より直感的なダッシュボードデザイン（調整中）
	•	✅ フィルタリング機能: ユーザーがデータを動的に検索できるようにする（開発中）
	•	🛠️ 多言語対応: 日本語 / 英語の翻訳機能（計画中）

🚀 インストールと起動方法

1. Backend のセットアップ

cd backend
pip install -r requirements.txt
python app.py

2. Frontend のセットアップ

cd frontend
npm install
npm start

✅ 主要な機能
	•	マルチ AI 連携: GPT-4, Claude, Gemini を活用した意思決定支援
	•	ナレッジベース: Weaviate によるデータ管理
	•	ダッシュボード: ユーザー向けデータ解析ツール
	•	リアルタイム処理: WebSocket ベースの応答（開発中）

📌 開発中の課題
	•	🛠️ セキュリティ強化: API 認証の導入（進行中）
	•	🛠️ レスポンス最適化: キャッシュの導入（計画中）
	•	🛠️ LLM の応答精度向上: LangChain の最適化（調整中）

