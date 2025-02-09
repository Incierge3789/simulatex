import asyncio
import json  # 追加
import logging
import os
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import google.auth
import motor.motor_asyncio
import requests
import weaviate
from bson import ObjectId
from communication_protocol import CommunicationProtocol
from dotenv import load_dotenv
from google.auth import transport
from google.auth.transport import requests
from google.oauth2 import id_token, service_account
from weaviate import WeaviateClient
from weaviate.auth import AuthApiKey, AuthClientCredentials
from weaviate.collections.classes.config import Configure, DataType, Property

logger = logging.getLogger(__name__)

load_dotenv()


# JSONエンコーダークラスの定義（クラス定義の前に配置）
class DateTimeEncoder(json.JSONEncoder):
    """datetime型をJSON形式に変換するためのエンコーダー"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)


class MongoJSONEncoder(json.JSONEncoder):
    """MongoDB特有の型をJSON形式に変換するためのエンコーダー"""

    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class DataManager:
    """
    データ管理クラス
    MongoDB と Weaviate を使用してトークンエコノミクスデータを管理する
    """

    def __init__(self):
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        self._weaviate_client = None
        self._access_token = None
        self.mongo_client = None
        self.mongo_db = None
        self.mongo_collection = None
        self.communication_protocol = None
        self.cache = {}
        self.default_token_data = {
            "token_symbol": "Unknown",
            "total_supply": "0",
            "circulating_supply": "0",
            "inflation_rate": "5%",
            "staking_reward": "0%",
        }

    async def _reset_attributes(self):
        """属性のリセット処理"""
        try:
            self._initialized = False
            if hasattr(self, "_weaviate_client") and self._weaviate_client is not None:
                try:
                    await self.close()
                except Exception as e:
                    logger.error(f"Error closing Weaviate client: {str(e)}")
                self._weaviate_client = None

            if self.mongo_client:
                try:
                    self.mongo_client.close()
                except Exception as e:
                    logger.error(f"Error closing MongoDB client: {str(e)}")
                self.mongo_client = None
                self.mongo_db = None
                self.mongo_collection = None

            self.communication_protocol = None
            logger.info("Attributes reset successfully")
        except Exception as e:
            logger.error(f"Error in _reset_attributes: {str(e)}")

    async def __aenter__(self):
        """非同期コンテキストマネージャのエントリーポイント"""
        try:
            await self.ensure_initialized()
            return self
        except Exception as e:
            logger.error(f"Error in __aenter__: {str(e)}")
            await self._reset_attributes()
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャの終了処理"""
        try:
            if self._weaviate_client:
                try:
                    self._weaviate_client.close()
                    logger.info("Weaviate client closed successfully")
                except Exception as e:
                    logger.error(f"Error closing Weaviate client: {str(e)}")

            await self._reset_attributes()
            logger.info("Context manager cleanup completed")
        except Exception as e:
            logger.error(f"Error in __aexit__: {str(e)}")
            raise

    async def _cleanup(self):
        """リソースのクリーンアップ処理"""
        try:
            if self.mongo_client:
                self.mongo_client.close()
                logger.info("MongoDB client closed")

            self._initialized = False
            self._weaviate_client = None
            self.mongo_client = None
            self.mongo_db = None
            self.mongo_collection = None
            logger.info("DataManager cleanup completed")
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")
            raise

    async def close(self):
        """接続のクローズ処理"""
        try:
            if self.mongo_client:
                try:
                    self.mongo_client.close()
                    logger.info("MongoDB connection closed")
                except Exception as e:
                    logger.error(f"Error closing MongoDB connection: {str(e)}")

            if self._weaviate_client:
                try:
                    self._weaviate_client.close()
                    logger.info("Weaviate connection closed")
                except Exception as e:
                    logger.error(f"Error closing Weaviate connection: {str(e)}")

            await self._reset_attributes()
            logger.info("All resources closed successfully")
        except Exception as e:
            logger.error(f"Error in close: {str(e)}")
            raise

    def __enter__(self):
        """同期コンテキストマネージャのエントリーポイント"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """同期コンテキストマネージャの終了処理"""
        self.close()

    def __del__(self):
        """デストラクタでのリソース解放"""
        try:
            self.close()
        except Exception as e:
            logger.error(f"Error in destructor: {str(e)}")

    def get_google_id_token(self):
        """Google IDトークンの取得"""
        try:
            # サービスアカウントの認証情報を読み込む
            credentials = service_account.IDTokenCredentials.from_service_account_file(
                os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                target_audience="299604111511-9mq6rv7k0mt6mh6ulr1k6tpqguaa480e.apps.googleusercontent.com",
            )

            # トークンの取得
            auth_req = requests.Request()
            credentials.refresh(auth_req)

            # IDトークンを取得
            self._access_token = credentials.token
            return self._access_token
        except Exception as e:
            logger.error(f"Failed to get Google ID token: {str(e)}")
            raise

    async def ensure_initialized(self):
        """初期化を確実に行うメソッド"""
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:
                return

            try:
                # MongoDB接続の初期化
                if os.path.exists("/.dockerenv"):
                    # Docker環境での接続
                    mongo_uri = "mongodb://mongo:27017/simulatex"
                    logger.info("Using Docker MongoDB configuration")
                else:
                    # ローカル環境での接続
                    mongo_uri = os.getenv(
                        "MONGO_URI", "mongodb://localhost:42042/simulatex"
                    )
                    logger.info("Using local MongoDB configuration")

                logger.info(f"Attempting MongoDB connection with URI: {mongo_uri}")

                mongo_options = {
                    "serverSelectionTimeoutMS": 30000,
                    "connectTimeoutMS": 30000,
                    "socketTimeoutMS": 30000,
                    "maxPoolSize": 50,
                    "retryWrites": True,
                    "retryReads": True,
                    "waitQueueTimeoutMS": 30000,
                    "directConnection": True,
                    "connect": True,
                }

                # MongoDB接続の確立とテスト
                retry_count = 0
                max_retries = 3
                retry_delay = 5

                while retry_count < max_retries:
                    try:
                        logger.info(
                            f"Attempting MongoDB connection (attempt {retry_count + 1}/{max_retries})"
                        )
                        self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
                            mongo_uri, **mongo_options
                        )
                        # 接続テスト
                        await self.mongo_client.admin.command("ping")
                        logger.info(
                            f"MongoDB connection successful on attempt {retry_count + 1}"
                        )

                        self.mongo_db = self.mongo_client.get_database("simulatex")
                        self.mongo_collection = self.mongo_db.get_collection(
                            "token_economics_data"
                        )
                        break
                    except Exception as mongo_error:
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.warning(
                                f"MongoDB connection attempt {retry_count} failed: {str(mongo_error)}, "
                                f"retrying in {retry_delay} seconds..."
                            )
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            logger.error(
                                f"MongoDB connection failed after {max_retries} attempts: {str(mongo_error)}"
                            )
                            raise

                # Weaviate接続の初期化
                # Weaviate接続の初期化
                try:
                    # イベントループの取得と設定
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    # 環境変数の検証ログ
                    logger.debug(
                        "Checking Weaviate configuration",
                        extra={
                            "has_weaviate_url": bool(os.getenv("WEAVIATE_URL")),
                            "has_weaviate_api_key": bool(os.getenv("WEAVIATE_API_KEY")),
                            "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
                            "environment": (
                                "docker" if os.path.exists("/.dockerenv") else "local"
                            ),
                        },
                    )

                    # トークンの取得と検証
                    try:
                        self._access_token = self.get_google_id_token()
                        logger.debug(
                            "Google ID token obtained",
                            extra={
                                "token_length": (
                                    len(self._access_token) if self._access_token else 0
                                ),
                                "token_type": type(self._access_token).__name__,
                            },
                        )
                    except Exception as token_error:
                        logger.error(
                            "Failed to obtain Google ID token",
                            extra={
                                "error": str(token_error),
                                "error_type": type(token_error).__name__,
                                "traceback": traceback.format_exc(),
                            },
                        )

                    # Weaviate Cloud接続の試行
                    weaviate_url = os.getenv("WEAVIATE_URL")
                    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

                    if weaviate_url and weaviate_api_key:
                        logger.info(
                            "Attempting Weaviate Cloud connection",
                            extra={
                                "url": weaviate_url,
                                "auth_type": "api_key",
                                "has_additional_headers": bool(
                                    os.getenv("OPENAI_API_KEY")
                                ),
                            },
                        )

                        auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_api_key)

                        try:
                            self._weaviate_client = weaviate.Client(
                                url=weaviate_url,
                                auth_client_secret=auth_config,
                                additional_headers={
                                    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
                                },
                            )

                            # 接続テスト
                            self._weaviate_client.schema.get()
                            logger.info("Successfully connected to Weaviate Cloud")

                        except Exception as cloud_conn_error:
                            logger.error(
                                "Weaviate Cloud connection failed",
                                extra={
                                    "error_type": type(cloud_conn_error).__name__,
                                    "error_message": str(cloud_conn_error),
                                    "response_status": getattr(
                                        cloud_conn_error, "status_code", None
                                    ),
                                    "response_body": getattr(
                                        cloud_conn_error, "response", None
                                    ),
                                    "traceback": traceback.format_exc(),
                                },
                            )
                            raise

                except Exception as cloud_error:
                    logger.warning(
                        "Failed to connect to Weaviate Cloud",
                        extra={
                            "error": str(cloud_error),
                            "error_type": type(cloud_error).__name__,
                            "traceback": traceback.format_exc(),
                        },
                    )

                    # ローカルWeaviateへのフォールバック処理
                    try:
                        weaviate_host = (
                            "weaviate" if os.path.exists("/.dockerenv") else "localhost"
                        )
                        weaviate_port = "8080"
                        weaviate_url = f"http://{weaviate_host}:{weaviate_port}"

                        logger.info(
                            "Attempting local Weaviate connection",
                            extra={
                                "url": weaviate_url,
                                "docker_env": os.path.exists("/.dockerenv"),
                                "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
                            },
                        )

                        self._weaviate_client = weaviate.Client(
                            url=weaviate_url,
                            additional_headers={
                                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
                            },
                        )

                        # 接続テスト
                        self._weaviate_client.schema.get()
                        logger.info("Successfully connected to local Weaviate")

                    except Exception as local_error:
                        logger.error(
                            "Local Weaviate connection failed",
                            extra={
                                "error_type": type(local_error).__name__,
                                "error_message": str(local_error),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        raise Exception(
                            "Failed to connect to both Cloud and local Weaviate"
                        )

                # コレクションのセットアップ
                await self._setup_weaviate_schema()

                # 初期化完了フラグを設定
                if (
                    self.mongo_collection is not None
                    and self._weaviate_client is not None
                ):
                    self._initialized = True
                    logger.info("DataManager initialized successfully")
                else:
                    raise Exception("Failed to initialize required services")

            except Exception as e:
                logger.error(f"Failed to initialize DataManager: {str(e)}")
                if hasattr(self, "mongo_client"):
                    self.mongo_client.close()
                self._initialized = False
                raise

    def _get_headers(self) -> Dict[str, str]:
        """認証ヘッダーの取得"""
        headers = {}
        if os.getenv("OPENAI_API_KEY"):
            headers["X-OpenAI-Api-Key"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("WEAVIATE_API_KEY"):
            headers["Authorization"] = f"Bearer {os.getenv('WEAVIATE_API_KEY')}"
        return headers

    def _check_network_status(self) -> Dict[str, Any]:
        """ネットワーク状態の確認"""
        try:
            weaviate_host = "weaviate" if os.path.exists("/.dockerenv") else "localhost"
            response = requests.get(f"http://{weaviate_host}:8080/v1/.well-known/ready")
            return {
                "status_code": response.status_code,
                "is_ready": response.status_code == 200,
                "response_body": response.text[:200],
            }
        except Exception as e:
            return {"error": str(e), "is_ready": False}

    @property
    def is_initialized(self) -> bool:
        """初期化状態を確認するプロパティ"""
        return self._initialized

    # 3. Weaviate関連のメソッド
    async def _setup_weaviate_schema(self):
        """Weaviateスキーマのセットアップ"""
        try:
            collection_name = "TokenEconomics"

            # v4でのスキーマ確認方法
            schema = self._weaviate_client.schema.get()
            collection_exists = any(
                cls["class"] == collection_name for cls in schema.get("classes", [])
            )

            if not collection_exists:
                # v4でのスキーマ作成
                schema_config = {
                    "class": collection_name,
                    "vectorizer": "text2vec-openai",
                    "properties": [
                        {
                            "name": "workflow_name",
                            "dataType": ["text"],
                        },
                        {
                            "name": "timestamp",
                            "dataType": ["text"],
                        },
                        {
                            "name": "overall_assessment",
                            "dataType": ["text"],
                        },
                        {
                            "name": "key_insights",
                            "dataType": ["text[]"],
                        },
                        {
                            "name": "risk_factors",
                            "dataType": ["text[]"],
                        },
                        {
                            "name": "recommendations",
                            "dataType": ["text[]"],
                        },
                        {
                            "name": "token_symbol",
                            "dataType": ["text"],
                        },
                        {
                            "name": "token_supply",
                            "dataType": ["number"],
                        },
                        {
                            "name": "total_supply",
                            "dataType": ["number"],
                        },
                        {
                            "name": "circulating_supply",
                            "dataType": ["number"],
                        },
                        {
                            "name": "inflation_rate",
                            "dataType": ["number"],
                        },
                        {
                            "name": "staking_reward",
                            "dataType": ["number"],
                        },
                    ],
                }

                self._weaviate_client.schema.create_class(schema_config)
                logger.info(f"Created {collection_name} collection")
            else:
                logger.info(f"{collection_name} collection already exists")

        except Exception as e:
            logger.error(f"Error setting up Weaviate schema: {str(e)}")
            raise

    @property
    def weaviate_client(self):
        """Weaviateクライアントの取得"""
        if self._weaviate_client is None:
            self._initialize()
        return self._weaviate_client

    async def save_to_weaviate(self, data: Dict[str, Any]) -> str:
        """Weaviateへのデータ保存"""
        try:
            save_data = {}
            for k, v in data.items():
                if k == "_id":
                    continue
                if isinstance(v, dict):
                    save_data[k] = self._remove_id_fields(v)
                elif isinstance(v, list):
                    save_data[k] = [
                        self._remove_id_fields(item) if isinstance(item, dict) else item
                        for item in v
                    ]
                else:
                    save_data[k] = v

            if "content" in save_data:
                if "summary" not in save_data["content"]:
                    save_data["content"]["summary"] = {}
                if "weighted_analysis" not in save_data["content"]["summary"]:
                    save_data["content"]["summary"]["weighted_analysis"] = {}
                save_data["content"]["summary"]["weighted_analysis"]["metadata"] = {
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0",
                    "analysis_type": "token_economics",
                    "status": "completed",
                }

            serialized_data = json.loads(json.dumps(save_data, cls=DateTimeEncoder))
            collection = self._weaviate_client.collections.get("TokenEconomics")
            result = collection.data.insert(serialized_data)
            return str(result.uuid)
        except Exception as e:
            logger.error(f"Error saving to Weaviate: {str(e)}")
            return ""

    def _remove_id_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """IDフィールドの削除"""
        if not isinstance(data, dict):
            return data
        result = {}
        for k, v in data.items():
            if k == "_id":
                continue
            if isinstance(v, dict):
                result[k] = self._remove_id_fields(v)
            elif isinstance(v, list):
                result[k] = [
                    self._remove_id_fields(item) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                result[k] = v
        return result

    async def search_weaviate(
        self, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Weaviateでの検索をBM25で実行"""
        try:
            if not self._weaviate_client:
                logger.error("Weaviate client not initialized")
                return []

            # BM25検索の実行
            response = (
                self._weaviate_client.query.get(
                    "TokenEconomics",
                    [
                        "workflow_name",
                        "timestamp",
                        "overall_assessment",
                        "key_insights",
                        "risk_factors",
                        "recommendations",
                        "token_symbol",
                        "token_supply",
                        "total_supply",
                        "circulating_supply",
                        "inflation_rate",
                        "staking_reward",
                    ],
                )
                .with_bm25(
                    query=query,
                    properties=[
                        "workflow_name",
                        "overall_assessment",
                        "key_insights",
                        "risk_factors",
                        "recommendations",
                        "token_symbol",
                    ],
                )
                .with_limit(limit)
                .do()
            )

            # レスポンスの処理
            if response and isinstance(response, dict):
                data = response.get("data", {})
                get_data = data.get("Get", {})
                results = get_data.get("TokenEconomics", [])
                if results is not None:
                    return results
            return []

        except Exception as e:
            logger.error(f"Error in search_weaviate: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    # 4. MongoDB関連のメソッド
    async def save_to_mongodb(self, data: Dict[str, Any]) -> str:
        """MongoDBにデータを保存し、保存されたドキュメントのIDを返す"""
        try:
            if not self.mongo_collection:
                await self.ensure_initialized()
                if not self.mongo_collection:
                    raise ValueError("MongoDB collection not initialized")

            # データの前処理
            processed_data = {
                **data,
                "timestamp": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
            }

            # MongoDBに保存
            result = await self.mongo_collection.insert_one(processed_data)

            if result.inserted_id:
                logger.info(
                    f"Successfully saved document to MongoDB with ID: {result.inserted_id}"
                )
                return str(result.inserted_id)
            else:
                raise Exception("Failed to save document to MongoDB")

        except Exception as e:
            logger.error(f"Error saving to MongoDB: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def get_from_mongodb(self, document_id: str) -> Dict[str, Any]:
        """MongoDBから指定されたIDのドキュメントを取得する"""
        try:
            if not self.mongo_collection:
                await self.ensure_initialized()
                if not self.mongo_collection:
                    raise ValueError("MongoDB collection not initialized")

            # ObjectIDに変換
            try:
                obj_id = ObjectId(document_id)
            except Exception as e:
                logger.error(f"Invalid MongoDB document ID format: {document_id}")
                raise ValueError(f"Invalid document ID format: {str(e)}")

            # ドキュメントの取得
            document = await self.mongo_collection.find_one({"_id": obj_id})

            if document:
                # ObjectIDをシリアライズ可能な形式に変換
                document = json.loads(json.dumps(document, cls=MongoJSONEncoder))
                logger.info(f"Successfully retrieved document: {document_id}")
                return document
            else:
                logger.warning(f"Document not found with ID: {document_id}")
                return None

        except ValueError as ve:
            logger.error(f"ValueError in get_from_mongodb: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Error retrieving from MongoDB: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def update_mongodb_document(
        self, document_id: str, update_data: Dict[str, Any]
    ) -> bool:
        """MongoDBのドキュメントを更新する"""
        try:
            if not self.mongo_collection:
                await self.ensure_initialized()
                if not self.mongo_collection:
                    raise ValueError("MongoDB collection not initialized")

            # ObjectIDに変換
            try:
                obj_id = ObjectId(document_id)
            except Exception as e:
                logger.error(f"Invalid MongoDB document ID format: {document_id}")
                raise ValueError(f"Invalid document ID format: {str(e)}")

            # 更新データの準備
            update_data["last_modified"] = datetime.now().isoformat()

            # ドキュメントの更新
            result = await self.mongo_collection.update_one(
                {"_id": obj_id}, {"$set": update_data}
            )

            if result.modified_count > 0:
                logger.info(f"Successfully updated document: {document_id}")
                return True
            else:
                logger.warning(f"Document not found or not modified: {document_id}")
                return False

        except ValueError as ve:
            logger.error(f"ValueError in update_mongodb_document: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Error updating MongoDB document: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def delete_from_mongodb(self, document_id: str) -> bool:
        """MongoDBからドキュメントを削除する"""
        try:
            if not self.mongo_collection:
                await self.ensure_initialized()
                if not self.mongo_collection:
                    raise ValueError("MongoDB collection not initialized")

            # ObjectIDに変換
            try:
                obj_id = ObjectId(document_id)
            except Exception as e:
                logger.error(f"Invalid MongoDB document ID format: {document_id}")
                raise ValueError(f"Invalid document ID format: {str(e)}")

            # ドキュメントの削除
            result = await self.mongo_collection.delete_one({"_id": obj_id})

            if result.deleted_count > 0:
                logger.info(f"Successfully deleted document: {document_id}")
                return True
            else:
                logger.warning(f"Document not found: {document_id}")
                return False

        except ValueError as ve:
            logger.error(f"ValueError in delete_from_mongodb: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Error deleting from MongoDB: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    # 5. データ操作関連のメソッド
    async def initialize_default_data(self):
        """デフォルトのトークンエコノミクスデータを初期化する"""
        try:
            default_data = {
                "workflow_name": "default_token_economics",
                "timestamp": datetime.now().isoformat(),
                "token_symbol": "DEFAULT",
                "token_supply": 1000000,
                "total_supply": 1000000,
                "circulating_supply": 500000,
                "initial_price": 1.0,
                "inflation_rate": 5,
                "staking_reward": 10,
                "overall_assessment": "Default token economics configuration",
                "key_insights": ["Initial configuration"],
                "risk_factors": ["Default risk assessment"],
                "recommendations": ["Default recommendations"],
            }

            exists = self._weaviate_client.collections.exists("TokenEconomics")
            if exists:
                collection = self._weaviate_client.collections.get("TokenEconomics")
                response = (
                    collection.query.hybrid(query="DEFAULT", alpha=0.5, limit=1)
                    .with_properties(["token_symbol"])
                    .do()
                )

                if not hasattr(response, "objects") or not response.objects:
                    await self.save_to_weaviate(default_data)
                    logger.info("Default token economics data initialized")
            else:
                await self.save_to_weaviate(default_data)
                logger.info(
                    "Default token economics data initialized in new collection"
                )

        except Exception as e:
            logger.error(f"Error initializing default data: {str(e)}")
            raise

    async def save_analysis_result(self, result: Dict[str, Any]) -> Dict[str, str]:
        """分析結果をMongoDBとWeaviateの両方に保存する"""
        try:
            # プロトコルメッセージの作成
            protocol_message = await self.communication_protocol.create_message(
                sender=result.get("sender", "System"),
                receiver=result.get("receiver", "User"),
                message_type="analysis_result",
                content=result,
            )

            # MongoDBに保存
            mongo_id = await self.save_to_mongodb(protocol_message)

            # Weaviateに保存
            weaviate_id = await self.save_to_weaviate(protocol_message)

            logger.info(
                f"Analysis result saved successfully. MongoDB ID: {mongo_id}, Weaviate ID: {weaviate_id}"
            )
            return {"mongodb_id": mongo_id, "weaviate_id": weaviate_id}

        except Exception as e:
            logger.error(f"Error saving analysis result: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def get_analysis_result(self, mongo_id: str) -> Dict[str, Any]:
        """MongoDBから分析結果を取得する"""
        try:
            document = await self.get_from_mongodb(mongo_id)
            if document:
                return document.get("content", {})
            logger.warning(f"No analysis result found for ID: {mongo_id}")
            return None

        except Exception as e:
            logger.error(f"Error retrieving analysis result: {str(e)}")
            return None

    async def find_similar_analyses(
        self, query_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """類似の分析結果を検索する"""
        try:
            # 入力データの型チェック
            if isinstance(query_data, str):
                logger.debug(f"Searching with string query: {query_data}")
                results = await self.search_weaviate(query=query_data)
                return results

            if isinstance(query_data, dict):
                # MongoDBの_idフィールドを除外
                search_data = {k: v for k, v in query_data.items() if k != "_id"}

                # クエリの作成前にtoken_symbolの存在確認
                query = search_data.get("token_symbol", "")
                logger.debug(f"Searching for token symbol: {query}")

                if not query:
                    logger.warning("No token symbol provided for search")
                    return []

                # Weaviateで検索を実行
                results = await self.search_weaviate(query=query)
                logger.debug(f"Search results: {results}")
                return results

            logger.error(f"Unsupported input type: {type(query_data)}")
            return []

        except Exception as e:
            logger.error(f"Error in find_similar_analyses: {str(e)}")
            return []

    # 6. フィードバックと検索関連のメソッド
    async def save_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """フィードバックをMongoDBに保存する"""
        try:
            if not self.mongo_collection:
                await self.ensure_initialized()
                if not self.mongo_collection:
                    raise ValueError("MongoDB collection not initialized")

            # プロトコルメッセージの作成
            protocol_message = await self.communication_protocol.create_message(
                sender="User",
                receiver="System",
                message_type="feedback",
                content=feedback_data,
            )

            # メタデータの追加
            protocol_message.update(
                {
                    "timestamp": datetime.now().isoformat(),
                    "feedback_type": feedback_data.get("type", "general"),
                    "status": "submitted",
                }
            )

            # MongoDBに保存
            result = await self.mongo_collection.insert_one(protocol_message)
            logger.info(f"Feedback saved successfully with ID: {result.inserted_id}")
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def save_search_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> str:
        """検索結果をMongoDBに保存する"""
        try:
            if not self.mongo_collection:
                await self.ensure_initialized()
                if not self.mongo_collection:
                    raise ValueError("MongoDB collection not initialized")

            # 検索結果のドキュメントを作成
            document = {
                "query": query,
                "results": results,
                "timestamp": datetime.now().isoformat(),
                "cache_expiry": (datetime.now() + timedelta(hours=24)).isoformat(),
                "result_count": len(results),
                "status": "cached",
            }

            # MongoDBに保存
            result = await self.mongo_collection.insert_one(document)
            logger.info(
                f"Search results saved successfully with ID: {result.inserted_id}"
            )
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Error saving search results: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def get_cached_search_results(self, query: str) -> List[Dict[str, Any]]:
        """過去24時間以内のキャッシュされた検索結果を取得する"""
        try:
            if not self.mongo_collection:
                await self.ensure_initialized()
                if not self.mongo_collection:
                    raise ValueError("MongoDB collection not initialized")

            # 24時間前の時刻を計算
            cache_time = datetime.now() - timedelta(hours=24)

            # キャッシュされた結果を検索
            result = await self.mongo_collection.find_one(
                {
                    "query": query,
                    "timestamp": {"$gte": cache_time.isoformat()},
                    "status": "cached",
                }
            )

            if result:
                logger.info(f"Found cached results for query: {query}")
                return result["results"]

            logger.info(f"No cached results found for query: {query}")
            return None

        except Exception as e:
            logger.error(f"Error getting cached search results: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    async def cache_search_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> None:
        """検索結果をキャッシュする"""
        try:
            if not self.mongo_collection:
                await self.ensure_initialized()
                if not self.mongo_collection:
                    raise ValueError("MongoDB collection not initialized")

            # 古いキャッシュを削除
            await self.mongo_collection.delete_many(
                {"query": query, "status": "cached"}
            )

            # 新しい結果をキャッシュ
            await self.save_search_results(query, results)
            logger.info(f"Search results cached successfully for query: {query}")

        except Exception as e:
            logger.error(f"Error caching search results: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def get_feedback_statistics(self) -> Dict[str, Any]:
        """フィードバックの統計情報を取得する"""
        try:
            if not self.mongo_collection:
                await self.ensure_initialized()
                if not self.mongo_collection:
                    raise ValueError("MongoDB collection not initialized")

            # 集計パイプライン
            pipeline = [
                {"$match": {"message_type": "feedback", "status": "submitted"}},
                {
                    "$group": {
                        "_id": "$feedback_type",
                        "count": {"$sum": 1},
                        "average_score": {"$avg": "$content.score"},
                    }
                },
            ]

            # 統計の集計
            stats = await self.mongo_collection.aggregate(pipeline).to_list(None)

            # 結果の整形
            result = {
                "total_feedback": sum(s["count"] for s in stats),
                "feedback_types": {
                    s["_id"]: {
                        "count": s["count"],
                        "average_score": round(s["average_score"], 2),
                    }
                    for s in stats
                },
                "last_updated": datetime.now().isoformat(),
            }

            logger.info("Feedback statistics retrieved successfully")
            return result

        except Exception as e:
            logger.error(f"Error getting feedback statistics: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    # 7. トークンエコノミクス関連のメソッド
    def get_default_token_data(self, token_symbol: str = "Unknown") -> Dict[str, Any]:
        """デフォルトのトークンデータを返す"""
        try:
            data = self.default_token_data.copy()
            data.update(
                {
                    "token_symbol": token_symbol,
                    "timestamp": datetime.now().isoformat(),
                    "workflow_name": "default_token_economics",
                    "overall_assessment": "Default token economics configuration",
                    "key_insights": ["Initial configuration"],
                    "risk_factors": ["Default risk assessment"],
                    "recommendations": ["Default recommendations"],
                }
            )
            return data
        except Exception as e:
            logger.error(f"Error in get_default_token_data: {str(e)}")
            return self.default_token_data

    async def get_token_economics_data(self, token_symbol: str) -> Dict[str, Any]:
        """トークンエコノミクスデータを取得する"""
        try:
            if not self._initialized:
                await self.ensure_initialized()

            # MongoDBコレクションの存在確認
            if self.mongo_collection is None:
                raise ValueError("MongoDB collection not initialized")

            # 新しいイベントループの取得と設定
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            try:
                # MongoDBからデータ取得
                query = {"token_symbol": token_symbol}
                result = await self.mongo_collection.find_one(query)

                if result:
                    logger.info(
                        f"Found token economics data in MongoDB for: {token_symbol}"
                    )
                    return result

                # Weaviateからのデータ取得を試行
                if self._weaviate_client is not None:
                    try:
                        weaviate_data = await loop.run_in_executor(
                            None,
                            lambda: self.fetch_token_data_from_weaviate(token_symbol),
                        )
                        if weaviate_data:
                            logger.info(
                                f"Found token economics data in Weaviate for: {token_symbol}"
                            )
                            return weaviate_data
                    except Exception as weaviate_error:
                        logger.error(
                            f"Weaviate data fetch error: {str(weaviate_error)}"
                        )

                logger.info(
                    f"No data found for token: {token_symbol}, returning default data"
                )
                return self.get_default_token_data(token_symbol)

            finally:
                # イベントループが新規作成された場合はクローズ
                if loop != asyncio.get_running_loop():
                    loop.close()

        except Exception as e:
            logger.error(f"Error fetching token economics data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self.get_default_token_data(token_symbol)

    async def save_token_economics_data(self, data: Dict[str, Any]) -> str:
        """トークンエコノミクスデータを保存する"""
        try:
            if not self._initialized:
                await self.ensure_initialized()

            # データの検証
            required_fields = ["token_symbol", "total_supply", "circulating_supply"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            # タイムスタンプの追加
            data["timestamp"] = datetime.now().isoformat()
            data["last_updated"] = datetime.now().isoformat()

            # MongoDBに保存
            result = await self.mongo_collection.insert_one(data)

            # キャッシュの更新
            self.cache[data["token_symbol"]] = (datetime.now(), data)

            # Weaviateにも保存
            await self.save_to_weaviate(data)

            logger.info(
                f"Token economics data saved successfully for: {data['token_symbol']}"
            )
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Error saving token economics data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def ensure_collection_exists(self) -> bool:
        try:
            if self._weaviate_client is None:
                await self.ensure_initialized()

            collections = self._weaviate_client.collections.get()
            exists = any(c.name == "TokenEconomics" for c in collections)

            if not exists:
                await self._setup_weaviate_schema()
                logger.info("Created TokenEconomics collection")
            else:
                logger.info("TokenEconomics collection already exists")

            return True
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            return False

    async def fetch_token_economics_data(self, query: str) -> List[Dict[str, Any]]:
        """トークンエコノミクスデータをBM25で検索"""
        try:
            await self.ensure_initialized()
            if self._weaviate_client is None:
                logger.error("Weaviate client not initialized")
                return []

            # BM25検索の実装
            response = (
                self._weaviate_client.query.get(
                    "TokenEconomics",
                    [
                        "workflow_name",
                        "timestamp",
                        "overall_assessment",
                        "key_insights",
                        "risk_factors",
                        "recommendations",
                        "token_symbol",
                        "token_supply",
                        "total_supply",
                        "circulating_supply",
                        "inflation_rate",
                        "staking_reward",
                    ],
                )
                .with_bm25(
                    query=query,
                    properties=[
                        "workflow_name",
                        "overall_assessment",
                        "key_insights",
                        "risk_factors",
                        "recommendations",
                        "token_symbol",
                    ],
                )
                .with_limit(10)
                .do()
            )

            # レスポンス処理の改善
            if not response:
                return []

            if not isinstance(response, dict):
                return []

            data = response.get("data", {})
            if not data:
                return []

            get_data = data.get("Get", {})
            if not get_data:
                return []

            results = get_data.get("TokenEconomics", [])
            if not results or not isinstance(results, list):
                return []

            return results

        except Exception as e:
            logger.error(f"Error fetching token economics data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    async def fetch_token_data_from_weaviate(self, token_symbol: str) -> Dict[str, Any]:
        """特定のトークンのデータをWeaviateからBM25で検索"""
        try:
            if not self._weaviate_client:
                return self.get_default_token_data(token_symbol)

            # BM25検索の実装
            response = (
                self._weaviate_client.query.get(
                    "TokenEconomics",
                    [
                        "workflow_name",
                        "timestamp",
                        "overall_assessment",
                        "key_insights",
                        "risk_factors",
                        "recommendations",
                        "token_symbol",
                        "token_supply",
                        "total_supply",
                        "circulating_supply",
                        "inflation_rate",
                        "staking_reward",
                    ],
                )
                .with_bm25(query=token_symbol, properties=["token_symbol"])
                .with_limit(1)
                .do()
            )

            # レスポンス処理の改善
            if isinstance(response, dict):
                data = response.get("data", {})
                get_data = data.get("Get", {})
                results = get_data.get("TokenEconomics", [])
                if results and isinstance(results, list) and len(results) > 0:
                    return results[0]

            return self.get_default_token_data(token_symbol)

        except Exception as e:
            logger.error(f"Error fetching token data from Weaviate: {str(e)}")
            return self.get_default_token_data(token_symbol)


async def main():
    # テスト用のコード
    data_manager = DataManager()

    # テストデータ
    test_data = {
        "workflow_name": "token_economics_analysis",
        "timestamp": datetime.now().isoformat(),
        "steps": [
            {"agent": "gpt4", "content": {"model_assessment": 85}},
            {
                "agent": "claude",
                "content": {"risk_assessment": [{"description": "High volatility"}]},
            },
            {
                "agent": "gemini",
                "content": {"key_metrics": ["Total supply: 1M", "Inflation rate: 5%"]},
            },
            {
                "agent": "analyst_ai",
                "content": {"recommendations": ["Implement staking rewards"]},
            },
        ],
        "summary": {
            "overall_assessment": "Model Assessment: 85/100",
            "key_insights": ["Total supply: 1M", "Inflation rate: 5%"],
            "risk_factors": ["High volatility"],
            "recommendations": ["Implement staking rewards"],
        },
    }

    # データの保存
    ids = await data_manager.save_analysis_result(test_data)
    print(
        f"Saved data. MongoDB ID: {ids['mongodb_id']}, Weaviate ID: {ids['weaviate_id']}"
    )

    # MongoDBからデータの取得
    retrieved_data = await data_manager.get_analysis_result(ids["mongodb_id"])
    print(f"Retrieved data from MongoDB: {retrieved_data}")

    # Weaviateで類似データの検索
    similar_data = await data_manager.find_similar_analyses(
        "token economics high volatility"
    )
    print(f"Similar data from Weaviate: {similar_data}")

    # フィードバックの保存
    feedback_id = await data_manager.save_feedback(
        {"score": 4, "comment": "Very helpful analysis"}
    )
    print(f"Saved feedback. MongoDB ID: {feedback_id}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
