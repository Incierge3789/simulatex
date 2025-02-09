import asyncio
import json
import logging
import os
import random
import re
import time
import traceback
from datetime import datetime  # 明示的なインポートを追加  # 正しいインポート
from typing import Any, Dict, List

import requests
from bson import ObjectId  # MongoDBのObjectId用
from communication_protocol import CommunicationProtocol
from data_manager import DataManager
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain.agents import AgentExecutor, AgentType, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

logging.basicConfig(
    filename="analyst_ai.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


class DateTimeEncoder(json.JSONEncoder):
    """日付とObjectIDのJSONエンコーディングを処理するカスタムエンコーダー"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, ObjectId):  # ObjectIdの処理を追加
            return str(obj)
        return super().default(obj)


class AnalystAI:
    def __init__(self):
        self.agent_name = "analyst_ai"
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
        # # DataManagerの初期化を遅延
        self.data_manager = None
        self.default_focus_area = "token_economics_analysis"
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        self.max_content_length = 4000  # GPT-4の制限を考慮
        self.max_text_length = 800  # 個別のテキストフィールドの制限
        self.max_list_items = 3  # リストの要素数制限

    async def ensure_initialized(self):
        """初期化の確認"""
        if self.data_manager is None:
            try:
                self.data_manager = DataManager()
                await self.data_manager.ensure_initialized()
                logger.info("DataManager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize DataManager: {str(e)}")
                raise

    def log_json(self, level, message, extra=None):
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
        }
        if extra:
            safe_extra = {}
            for key, value in extra.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    safe_extra[key] = value
                else:
                    safe_extra[key] = str(value)
            log_data.update(safe_extra)
        logger.log(getattr(logging, level), json.dumps(log_data, default=str))

    def generate_unique_prompt(self):
        focus_areas = [
            "market trend analysis",
            "competitive landscape assessment",
            "token utility and adoption potential",
            "economic model sustainability",
            "regulatory compliance and risk assessment",
            "technological innovation and scalability",
        ]
        selected_focus = random.choice(focus_areas)
        return f"Focus on analyzing the token economics model with emphasis on {selected_focus}."

    def generate_context_based_description(self, field, value):
        if field == "token_supply":
            return f"The total supply of tokens is {value}, which impacts the token's scarcity and potential market dynamics."
        elif field == "initial_price":
            return f"The initial token price is set at {value}, which serves as a baseline for future price movements and investor expectations."
        elif field == "inflation_rate":
            return f"The inflation rate of {value} affects the token's long-term value and distribution strategy."
        elif field == "staking_reward":
            return f"Staking rewards of {value} incentivize token holders to participate in network security and governance."
        else:
            return f"The {field} is {value}, which is a key factor in the token economics model."

    async def search_wrapper(self, query: str) -> Dict[str, Any]:
        """非同期検索ラッパー関数"""
        try:
            results = await self.search(query)
            return results
        except Exception as e:
            self.log_json(
                "ERROR", "Search wrapper error", {"error": str(e), "query": query}
            )
            return {"error": str(e)}

    def _limit_dict_size(self, d: Dict) -> Dict:
        """辞書の値のサイズを制限"""
        limited_dict = {}
        for k, v in d.items():
            if isinstance(v, str) and len(v) > self.max_text_length:
                limited_dict[k] = v[: self.max_text_length] + "..."
            elif isinstance(v, list):
                limited_dict[k] = v[: self.max_list_items]
            elif isinstance(v, dict):
                limited_dict[k] = self._limit_dict_size(v)
            else:
                limited_dict[k] = v
        return limited_dict

    def validate_and_enrich_input_data(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """入力データの検証と最適化"""
        try:
            # 必須フィールドの定義
            essential_fields = [
                "token_symbol",
                "token_supply",
                "initial_price",
                "inflation_rate",
                "staking_reward",
            ]

            # 必須フィールドのみを保持
            filtered_data = {k: input_data.get(k, "Unknown") for k in essential_fields}

            # 追加フィールドの選択的な保持
            allowed_additional_fields = [
                "user_message",
                "previous_responses",
                "similar_analyses",
            ]

            for field in allowed_additional_fields:
                if field in input_data:
                    filtered_data[field] = input_data[field]

            # 類似分析の制限
            if "similar_analyses" in filtered_data:
                filtered_data["similar_analyses"] = filtered_data["similar_analyses"][
                    :1
                ]

            # 長いテキストフィールドの制限
            for key, value in filtered_data.items():
                if isinstance(value, str):
                    if len(value) > self.max_text_length:
                        filtered_data[key] = value[: self.max_text_length] + "..."
                elif isinstance(value, list):
                    filtered_data[key] = value[: self.max_list_items]
                elif isinstance(value, dict):
                    filtered_data[key] = self._limit_dict_size(value)

            # 全体のサイズチェック
            total_content = json.dumps(filtered_data)
            if len(total_content) > self.max_content_length:
                # さらなる圧縮が必要な場合
                for key, value in filtered_data.items():
                    if isinstance(value, str):
                        filtered_data[key] = value[: self.max_text_length // 2] + "..."

            # デバッグログの追加
            self.log_json(
                "DEBUG",
                "Input data validation complete",
                {
                    "original_size": len(str(input_data)),
                    "filtered_size": len(str(filtered_data)),
                    "similar_analyses_count": len(
                        filtered_data.get("similar_analyses", [])
                    ),
                    "keys": list(filtered_data.keys()),
                    "total_tokens_estimate": len(total_content) // 4,  # 概算
                },
            )

            return filtered_data

        except Exception as e:
            self.log_json(
                "ERROR",
                "Error in validate_and_enrich_input_data",
                {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "input_data_keys": list(input_data.keys()) if input_data else None,
                },
            )
            return input_data

    def _limit_dict_size(self, d: Dict) -> Dict:
        """辞書の値のサイズを制限"""
        limited_dict = {}
        for k, v in d.items():
            if isinstance(v, str) and len(v) > 1000:
                limited_dict[k] = v[:1000] + "..."
            elif isinstance(v, list):
                limited_dict[k] = v[:5]
            elif isinstance(v, dict):
                limited_dict[k] = self._limit_dict_size(v)
            else:
                limited_dict[k] = v
        return limited_dict

    async def get_token_economics_data(self, token_symbol: str) -> Dict[str, Any]:
        """トークンエコノミクスデータを取得"""
        try:
            # # 初期化の確認
            await self.ensure_initialized()

            if self.data_manager is None:
                raise ValueError("DataManager is not initialized")

            # # 非同期処理を直接実行
            data = await self.data_manager.get_token_economics_data(token_symbol)
            return data

        except Exception as e:
            logger.error(f"Error fetching token economics data: {str(e)}")
            return self.get_default_token_data(token_symbol)

    def get_default_token_data(self, token_symbol: str) -> Dict[str, Any]:
        """
        デフォルトのトークンデータを返す
        """
        return {
            "token_symbol": token_symbol,
            "token_supply": "Unknown",
            "token_supply_description": "Data for token_supply is missing",
            "initial_price": "Unknown",
            "initial_price_description": "Data for initial_price is missing",
            "inflation_rate": "Unknown",
            "inflation_rate_description": "Data for inflation_rate is missing",
            "staking_reward": "Unknown",
            "staking_reward_description": "Data for staking_reward is missing",
            "timestamp": datetime.now().isoformat(),  # datetime.datetimeではなくdatetimeを使用
        }

    async def serp_search(self, query):
        try:
            self.log_json("DEBUG", f"Starting SerpAPI search for query: {query}")
            search = SerpAPIWrapper(serpapi_api_key=self.SERPAPI_API_KEY)
            results = await search.arun(query)
            self.log_json("INFO", f"SerpAPI search successful for query: {query}")
            return results
        except Exception as e:
            self.log_json(
                "ERROR",
                f"SerpAPI search failed: {str(e)}",
                {"exception": str(e), "traceback": traceback.format_exc()},
            )
            return None

    async def google_search(self, query):
        try:
            self.log_json("DEBUG", f"Starting Google Custom Search for query: {query}")
            service = build("customsearch", "v1", developerKey=self.GOOGLE_API_KEY)
            res = await asyncio.to_thread(
                service.cse().list(q=query, cx=self.GOOGLE_CSE_ID).execute
            )
            self.log_json("INFO", f"Google Custom Search successful for query: {query}")
            return [
                {
                    "title": item["title"],
                    "snippet": item["snippet"],
                    "link": item["link"],
                }
                for item in res.get("items", [])
            ]
        except HttpError as e:
            self.log_json(
                "ERROR",
                f"Google Custom Search HTTP error: {str(e)}",
                {"exception": str(e), "traceback": traceback.format_exc()},
            )
            return None
        except Exception as e:
            self.log_json(
                "ERROR",
                f"Google Custom Search failed: {str(e)}",
                {"exception": str(e), "traceback": traceback.format_exc()},
            )
            return None

    async def save_search_results(self, query: str, results: list):
        """
        検索結果をデータベースに保存する
        """
        try:
            await self.data_manager.save_search_results(query, results)
            self.log_json("INFO", f"Search results saved for query: {query}")
        except Exception as e:
            self.log_json("ERROR", f"Failed to save search results: {str(e)}")

    async def search(self, query):
        """
        検索を実行し、結果をキャッシュする
        """
        self.log_json("DEBUG", f"Starting search function for query: {query}")
        # キャッシュから結果を取得
        cached_results = await self.data_manager.get_cached_search_results(query)
        if cached_results:
            self.log_json("INFO", f"Using cached results for query: {query}")
            return cached_results

        # キャッシュがない場合、新たに検索を実行
        results = await self.serp_search(query)
        if results is None:
            self.log_json(
                "WARNING", "SerpAPI search failed, falling back to Google Custom Search"
            )
            results = await self.google_search(query)

        if results:
            # 検索結果を保存
            await self.save_search_results(query, results)
            # キャッシュに結果を保存
            await self.data_manager.cache_search_results(query, results)
        else:
            self.log_json("ERROR", "Both SerpAPI and Google Custom Search failed")
            return []

        return results

    def set_default_values(self, data):
        defaults = {
            "description": "No description provided",
            "recommendation": "No specific recommendation",
            "impact_score": 0,
            "feasibility_score": 0,
        }
        for key, value in defaults.items():
            if key not in data or data[key] is None:
                data[key] = value
        for field in [
            "key_metrics",
            "risk_factors",
            "optimization_strategies",
            "innovation_proposals",
        ]:
            if field in data and isinstance(data[field], list):
                for item in data[field]:
                    if isinstance(item, dict):
                        if "description" not in item or item["description"] is None:
                            item["description"] = "No description provided"
        return data

    def parse_openai_response(self, response_text: str) -> Dict[str, Any]:
        """OpenAIレスポンスのパース処理"""
        try:
            # 1. ReActフォーマットの除去
            cleaned_text = re.sub(
                r"Thought:.*?Action:.*?Action Input:.*?Observation:",
                "",
                response_text,
                flags=re.DOTALL,
            )

            # 2. JSON部分の抽出
            json_match = re.search(r"(\{[\s\S]*\})", cleaned_text)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    # 特殊文字の処理
                    json_str = re.sub(r"[\n\r\t]", " ", json_str)
                    json_str = re.sub(r'\\(?!["\\/bfnrt])', "", json_str)
                    parsed_response = json.loads(json_str)
                    if self.validate_response(parsed_response):
                        return parsed_response
                except json.JSONDecodeError:
                    self.log_json("DEBUG", "JSON extraction failed")

            # 3. 構造化テキスト解析を試みる
            structured_response = self.structure_text_response(response_text)
            if self.validate_response(structured_response):
                return structured_response

            # 4. フォールバックレスポンス（意味のあるデフォルト値）
            return {
                "model_assessment": 50,  # 中間値
                "key_metrics": [{"description": "Default Metric", "value": "Unknown"}],
                "risk_factors": [
                    {"description": "Unable to analyze risks", "impact": "medium"}
                ],
                "optimization_strategies": [
                    {"description": "Further analysis needed", "priority": "high"}
                ],
                "market_fit_analysis": {
                    "score": 50,
                    "justification": "Analysis incomplete",
                    "improvement_suggestions": ["Retry analysis with more data"],
                },
            }

        except Exception as e:
            self.log_json("ERROR", "Error in parse_openai_response", {"error": str(e)})
            return self.fallback_response()

    def validate_response(self, response: Dict[str, Any]) -> bool:
        """レスポンスの検証"""
        try:
            # 必須フィールドの検証
            required_fields = {
                "model_assessment": int,
                "key_metrics": list,
                "risk_factors": list,
                "market_fit_analysis": dict,
            }

            for field, field_type in required_fields.items():
                if field not in response:
                    return False
                if not isinstance(response[field], field_type):
                    return False

            # 配列の検証
            for field in ["key_metrics", "risk_factors"]:
                if not response.get(field) or not all(
                    isinstance(item, dict) and "description" in item
                    for item in response[field]
                ):
                    return False

            # market_fit_analysisの検証
            market_fit = response.get("market_fit_analysis", {})
            if not all(k in market_fit for k in ["score", "justification"]):
                return False

            return True

        except Exception:
            return False

    def structure_text_response(self, text):
        """テキストレスポンスを構造化データに変換"""
        try:
            structured_response = {
                "model_assessment": 0,
                "key_metrics": [],
                "risk_factors": [],
                "optimization_strategies": [],
                "innovation_proposals": [],
                "market_fit_analysis": {
                    "score": 0,
                    "justification": "",
                    "improvement_suggestions": [],
                },
            }

            # テキストの前処理とクリーニング
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            current_section = None

            for line in lines:
                # モデル評価スコアの抽出（より柔軟な数字の抽出）
                if (
                    "model_assessment" in line.lower()
                    or "model assessment" in line.lower()
                ):
                    match = re.search(r"(\d+)", line)
                    if match:
                        structured_response["model_assessment"] = int(match.group(1))
                    continue

                # セクションヘッダーの検出（より柔軟なマッチング）
                elif any(
                    section in line.lower()
                    for section in [
                        "key metrics",
                        "key_metrics",
                        "risk factors",
                        "risk_factors",
                        "optimization strategies",
                        "optimization_strategies",
                        "innovation proposals",
                        "innovation_proposals",
                        "market fit analysis",
                        "market_fit_analysis",
                    ]
                ):
                    # セクション名の正規化
                    section_map = {
                        "key metrics": "key_metrics",
                        "risk factors": "risk_factors",
                        "optimization strategies": "optimization_strategies",
                        "innovation proposals": "innovation_proposals",
                        "market fit analysis": "market_fit_analysis",
                    }
                    for key in section_map:
                        if key in line.lower():
                            current_section = section_map[key]
                            break
                    continue

                # セクション内容の処理
                elif current_section and line:
                    if current_section in [
                        "key_metrics",
                        "risk_factors",
                        "optimization_strategies",
                        "innovation_proposals",
                    ]:
                        # 箇条書きと引用符の処理
                        line = line.strip('"-*•[] ')

                        # 説明とスコアの分離
                        if ":" in line:
                            key, value = line.split(":", 1)
                            item = {"description": value.strip(), "type": key.strip()}
                        else:
                            item = {"description": line}

                        structured_response[current_section].append(item)

                    elif current_section == "market_fit_analysis":
                        if "score" in line.lower():
                            match = re.search(r"(\d+)", line)
                            if match:
                                structured_response["market_fit_analysis"]["score"] = (
                                    int(match.group(1))
                                )
                        elif "justification" in line.lower():
                            value = (
                                line.split(":", 1)[1].strip() if ":" in line else line
                            )
                            structured_response["market_fit_analysis"][
                                "justification"
                            ] = value
                        elif (
                            "improvement" in line.lower()
                            or "suggestions" in line.lower()
                        ):
                            value = (
                                line.split(":", 1)[1].strip() if ":" in line else line
                            )
                            structured_response["market_fit_analysis"][
                                "improvement_suggestions"
                            ].append(value)

            # レスポンスの検証
            self.log_json(
                "DEBUG",
                "Text response structured successfully",
                {
                    "sections": list(structured_response.keys()),
                    "metrics_count": len(structured_response["key_metrics"]),
                    "risks_count": len(structured_response["risk_factors"]),
                    "has_market_analysis": bool(
                        structured_response["market_fit_analysis"]["justification"]
                    ),
                },
            )

            return structured_response

        except Exception as e:
            self.log_json(
                "ERROR",
                "Error in structure_text_response",
                {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "text_preview": text[:200] if text else None,
                },
            )
            return self.fallback_response()

    async def query_analyst_ai(self, message: dict):
        """Analyst AIのクエリ処理"""
        start_time = time.time()
        receiver = message.get("sender", "Unknown")

        try:
            await self.ensure_initialized()
            self.log_json(
                "INFO",
                "Starting Analyst AI query",
                {
                    "message_type": message.get("message_type"),
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # 入力データの検証と最適化
            input_data = message.get("content", {})
            if not input_data:
                raise ValueError("Empty input data")

            # 入力データのサイズを記録
            self.log_json(
                "DEBUG",
                "Input data size before validation",
                {"size": len(str(input_data))},
            )

            input_data = self.validate_and_enrich_input_data(input_data)

            # トークンエコノミクスデータの取得と最適化
            token_symbol = input_data.get("token_symbol", "Unknown")
            token_data = await self.get_token_economics_data(token_symbol)
            input_data.update(token_data)

            # ObjectIdを文字列に変換
            if "_id" in input_data:
                input_data["_id"] = str(input_data["_id"])

            # LLMの初期化（より厳格なトークン制限）
            llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0,
                max_tokens=1500,  # 出力トークンをさらに制限
                request_timeout=300,
                max_retries=2,
            )

            # 入力データの制限
            input_data = self.validate_and_enrich_input_data(input_data)

            # 検索ツールの設定
            serp_tool = Tool(
                name="Search",
                func=self.search_wrapper,
                description="Search for market trends and token economics information",
            )

            # 過去の類似分析を1件のみに制限
            similar_analyses = await self.data_manager.find_similar_analyses(
                json.dumps(input_data, cls=DateTimeEncoder)
            )
            similar_analyses = similar_analyses[:1]  # 1件のみに制限

            # ユニークなフォーカスエリアの生成
            unique_focus = self.generate_unique_prompt()

            # プロンプトテンプレートの修正
            prompt = PromptTemplate.from_template(
                """You are an AI market trend and token economics analyst. Analyze the following data and provide a JSON response.

                Input data:
                {input}

                Tools available: {tools}
                Tool Names: {tool_names}

                RESPONSE JSON FORMAT:
                {{
                    "model_assessment": <integer 0-100>,
                    "key_metrics": [
                        {{
                            "description": "<specific metric name>",
                            "value": "<actual value>"
                        }}
                    ],
                    "risk_factors": [
                         {{
                            "description": "<specific risk>",
                            "impact": "high/medium/low"
                        }}
                    ],
                    "optimization_strategies": [
                        {{
                            "description": "<specific strategy>",
                            "priority": "high/medium/low"
                        }}
                    ],
                    "market_fit_analysis": {{
                        "score": <integer 0-100>,
                        "justification": "<clear explanation>",
                        "improvement_suggestions": [
                            "<actionable suggestion>"
                        ]
                    }}
                }}

                Process:
                1. First, analyze the input data
                2. If needed, use available tools to gather more information
                3. Format your final analysis as JSON

                Rules:
                1. Response must be ONLY valid JSON
                2. All scores must be 0-100
                3. Each array needs at least one item
                4. No text outside JSON structure

                {agent_scratchpad}
                """
            )

            # エージェントの作成（プロンプトを直接使用）
            agent = create_react_agent(
                llm=llm,
                tools=[serp_tool],
                prompt=prompt,  # プロンプトテンプレートを直接使用
            )

            # エージェントエグゼキューターの作成
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=[serp_tool],
                max_iterations=5,
                handle_parsing_errors=True,  # パースエラーのハンドリングを有効化
                early_stopping_method="force",  # 強制的な停止を有効化
                agent_kwargs={
                    "input": json.dumps(input_data),
                    "tools": str([serp_tool]),
                    "tool_names": ", ".join([tool.name for tool in [serp_tool]]),
                    "agent_scratchpad": "",
                },
            )

            self.log_json(
                "DEBUG",
                "Executing agent",
                {
                    "input_data_length": len(str(input_data)),
                    "similar_analyses_count": len(similar_analyses),
                },
            )

            # エージェントの実行と応答処理
            try:
                # handle_parsing_errorsを有効化
                agent_executor = AgentExecutor.from_agent_and_tools(
                    agent=agent,
                    tools=[serp_tool],
                    handle_parsing_errors=True,  # パースエラーのハンドリングを有効化
                    max_iterations=5,  # イテレーション数を制限
                    early_stopping_method="force",  # 強制的な停止を有効化
                    agent_kwargs={
                        "tool_names": ", ".join([tool.name for tool in [serp_tool]]),
                        "agent_scratchpad": "",
                    },
                )

                # エージェントの実行
                response = await agent_executor.ainvoke(
                    {"input": json.dumps(input_data)}
                )

                if not isinstance(response, dict) or "output" not in response:
                    self.log_json(
                        "WARNING",
                        "Unexpected response format",
                        {"response_type": str(type(response))},
                    )
                    return self.fallback_response()

                # レスポンスの処理
                try:
                    analyst_response = self.parse_openai_response(response["output"])
                    if not analyst_response:
                        self.log_json(
                            "WARNING", "Empty response from parse_openai_response"
                        )
                        return self.fallback_response()

                except Exception as parse_error:
                    self.log_json(
                        "ERROR",
                        "Error processing OpenAI response",
                        {
                            "error": str(parse_error),
                            "response_length": len(str(response.get("output", ""))),
                        },
                    )
                    return self.fallback_response()

                # レスポンスの標準化
                analyst_response = self.set_default_values(analyst_response)
                execution_time = time.time() - start_time
                standardized_response = self.standardize_response(analyst_response)
                standardized_response.update(
                    {
                        "execution_time": execution_time,
                        "focus_area": unique_focus,
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "agent_name": self.agent_name,
                            "response_type": "analysis",
                        },
                    }
                )

                self.log_json(
                    "INFO",
                    "Query executed successfully",
                    {
                        "execution_time": execution_time,
                        "response_size": len(str(standardized_response)),
                        "focus_area": unique_focus,
                    },
                )

                return standardized_response

            except Exception as exec_error:
                self.log_json(
                    "ERROR",
                    "Agent execution error",
                    {
                        "error": str(exec_error),
                        "error_type": type(exec_error).__name__,
                        "traceback": traceback.format_exc(),
                    },
                )
                return await self.handle_error(
                    exec_error,
                    receiver,
                    start_time,
                    input_data,
                    "AGENT_EXECUTION_ERROR",
                )

        except Exception as e:
            return await self.handle_error(
                e, receiver, start_time, input_data, "ANALYST_AI_PROCESSING_ERROR"
            )

    def standardize_response(self, response):
        standardized = {}
        for key, value in response.items():
            if isinstance(value, dict):
                standardized[key] = self.standardize_response(value)
            elif isinstance(value, list):
                standardized[key] = [
                    self.standardize_response(item) if isinstance(item, dict) else item
                    for item in value
                ]
            elif isinstance(value, (int, float)):
                standardized[key] = float(value)
            elif isinstance(value, str):
                standardized[key] = value.strip()
            else:
                standardized[key] = str(value)
        return standardized

    def fallback_response(self):
        return {
            "model_assessment": 0,
            "key_metrics": [],
            "risk_factors": [],
            "optimization_strategies": [],
            "innovation_proposals": [],
            "market_fit_analysis": {
                "score": 0,
                "justification": "Unable to generate analysis due to an error",
                "improvement_suggestions": [],
            },
        }

    async def handle_error(
        self,
        e: Exception,
        receiver: str,
        start_time: float,
        input_data: Dict[str, Any],
        error_code: str,
    ) -> Dict[str, Any]:
        """エラーハンドリング処理の改善"""
        error_message = f"Error in Analyst AI processing: {str(e)}"
        self.log_json(
            "ERROR",
            error_message,
            {
                "traceback": traceback.format_exc(),
                "input_data_preview": str(input_data)[:200] if input_data else None,
            },
        )

        execution_time = time.time() - start_time
        partial_results = self.extract_partial_results(input_data)

        error_response = {
            "error_code": error_code,
            "error_message": error_message,
            "execution_time": execution_time,
            "partial_results": partial_results,
            "focus_area": self.default_focus_area,
        }

        try:
            protocol = CommunicationProtocol()
            return await protocol.create_message(
                sender=self.agent_name,
                receiver=receiver,
                message_type="error",
                content=error_response,
            )
        except Exception as comm_error:
            self.log_json(
                "ERROR", "Communication protocol error", {"error": str(comm_error)}
            )
            return error_response

    def extract_partial_results(self, data):
        """部分的な結果を抽出し、シリアライズ可能な形式に変換"""
        if isinstance(data, dict):
            return {
                k: self.extract_partial_results(v)
                for k, v in data.items()
                if not isinstance(v, (dict, list))
                or k in ["token_symbol", "focus_area"]
            }
        elif isinstance(data, list):
            return [
                self.extract_partial_results(item)
                for item in data
                if not isinstance(item, (dict, list))
            ]
        elif isinstance(data, ObjectId):
            return str(data)
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data


# メイン関数も非同期に変更
async def main():
    agent = AnalystAI()
    test_input = {
        "token_supply": 1000000,
        "initial_price": 0.1,
        "inflation_rate": 0.05,
        "staking_reward": 0.08,
        "burn_mechanism": "2% of transaction fees",
        "vesting_schedule": "10% unlock per quarter",
    }
    test_message = await CommunicationProtocol.create_message(
        sender="System",
        receiver="analyst_ai",
        message_type="token_economics_data",
        content=test_input,
    )
    result = await agent.query_analyst_ai(test_message)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
