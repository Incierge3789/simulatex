import asyncio
import datetime
import json
import logging
import os
import random
import re
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List

import google.generativeai as genai
from communication_protocol import CommunicationProtocol
from dotenv import load_dotenv

# ロギングの設定
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("gemini_agent.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class GeminiAgent:
    def __init__(self):
        self.agent_name = "gemini"
        self.default_focus_area = "token_economics_analysis"  # 追加
        self.communication_protocol = CommunicationProtocol()
        self.model = genai.GenerativeModel("gemini-1.5-pro")

    def log_json(self, level: str, message: str, extra: Dict = None) -> None:

        try:
            # 基本的なログデータの構造
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "message": message,
                "agent_name": self.agent_name,
            }

            # extraデータの安全な処理
            if extra:
                safe_extra = {}
                for key, value in extra.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        safe_extra[key] = value
                    else:
                        try:
                            safe_extra[key] = str(value)
                        except Exception as e:
                            safe_extra[key] = (
                                f"<未シリアル化の値: {type(value).__name__}>"
                            )
                log_data.update(safe_extra)

            # ログレベルの検証
            valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            log_level = level.upper()
            if log_level not in valid_levels:
                log_level = "ERROR"
                log_data["original_level"] = level

            # ログの出力
            logger.log(
                getattr(logging, log_level),
                json.dumps(log_data, default=str, ensure_ascii=False),
            )

        except Exception as e:
            # ログ出力自体のエラーをフォールバック処理
            fallback_data = {
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "message": "Logging error",
                "error": str(e),
                "original_message": message,
            }
            logger.error(json.dumps(fallback_data, default=str, ensure_ascii=False))

    def create_error_response(self, error_message: str) -> dict:
        """エラーレスポンスを生成する"""
        return {
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
            "agent_name": self.agent_name,
        }

    def generate_unique_prompt(self):
        focus_areas = [
            "short-term growth potential",
            "market volatility resilience",
            "community engagement and governance",
            "interoperability with other DeFi protocols",
            "regulatory compliance and adaptability",
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

    def validate_and_enrich_input_data(self, input_data):
        required_fields = [
            "token_supply",
            "initial_price",
            "inflation_rate",
            "staking_reward",
        ]
        for field in required_fields:
            if field not in input_data or input_data[field] is None:
                input_data[field] = "Unknown"
                input_data[f"{field}_description"] = (
                    f"Data for {field} is missing. This may affect the accuracy of the analysis."
                )
            else:
                input_data[f"{field}_description"] = (
                    self.generate_context_based_description(field, input_data[field])
                )
        return input_data

    def standardize_data(self, data, depth=0):
        if depth > 10:  # Prevent infinite recursion
            return str(data)
        if isinstance(data, dict):
            return {k: self.standardize_data(v, depth + 1) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.standardize_data(item, depth + 1) for item in data]
        elif isinstance(data, (int, float)):
            return float(data)
        elif isinstance(data, str):
            return data.strip()
        else:
            return str(data)

    def set_default_values(self, data):
        # 改善: データがNoneの場合のチェックを追加
        if data is None:
            self.log_json("ERROR", "Received None data in set_default_values")
            return {}  # 空の辞書を返す

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

    async def query_gemini(self, message: dict):
        start_time = time.time()
        receiver = message.get("sender", "Unknown")
        input_data = {}  # 初期化を追加

        try:
            self.log_json(
                "INFO",
                "Starting Gemini query",
                {"message_type": message.get("message_type")},
            )
            input_data = message.get("content", {})
            input_data = self.validate_and_enrich_input_data(input_data)
            input_data = self.standardize_data(input_data)
            unique_focus = self.generate_unique_prompt()
            prompt = self.generate_analysis_prompt(input_data, unique_focus)
            self.log_json(
                "DEBUG", "Sending request to Gemini API", {"prompt_length": len(prompt)}
            )

            try:
                response = await asyncio.to_thread(self.model.generate_content, prompt)
                self.log_json(
                    "DEBUG",
                    "Received response from Gemini API",
                    {"response_length": len(response.text)},
                )
            except Exception as api_error:
                self.log_json(
                    "ERROR", "Error calling Gemini API", {"error": str(api_error)}
                )
                raise

            try:
                gemini_response = self.parse_gemini_response(response.text)
            except json.JSONDecodeError as json_error:
                self.log_json(
                    "ERROR",
                    "Error decoding JSON response from Gemini API",
                    {"error": str(json_error)},
                )
                gemini_response = self.extract_structured_data(response.text)

            self.log_json("INFO", "Gemini query completed successfully")
            standardized_response = self.process_gemini_response(
                gemini_response, unique_focus
            )
            execution_time = time.time() - start_time
            standardized_response["execution_time"] = execution_time

            # create_message
            return await self.communication_protocol.create_message(
                sender=self.agent_name,
                receiver=receiver,
                message_type="token_economics_data_analysis",
                content=standardized_response,
            )

        except Exception as e:
            return await self.handle_error(e, receiver, start_time, input_data)

    def parse_gemini_response(self, response_text: str) -> dict:
        """Gemini APIからの応答をパースし、構造化データを返す"""
        try:
            # デバッグログの追加
            self.log_json(
                "DEBUG",
                "Processing Gemini response",
                {
                    "response_length": len(response_text),
                    "response_preview": response_text[:200],
                },
            )

            # 1. JSONブロックの抽出と解析
            json_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)

            if json_matches:
                for json_str in json_matches:
                    try:
                        # JSONの前後の空白を除去し、余分な文字を削除
                        cleaned_json = json_str.strip()
                        cleaned_json = re.sub(r",\s*}", "}", cleaned_json)
                        cleaned_json = re.sub(r",\s*]", "]", cleaned_json)

                        parsed_data = json.loads(cleaned_json)
                        self.log_json(
                            "DEBUG",
                            "Successfully parsed JSON block",
                            {"parsed_keys": list(parsed_data.keys())},
                        )
                        return parsed_data
                    except json.JSONDecodeError:
                        continue

            # 2. 構造化データの抽出
            self.log_json(
                "WARNING",
                "No valid JSON found, attempting structured data extraction",
                {
                    "text_length": len(response_text),
                    "text_preview": response_text[:200],
                },
            )

            # セクション分割による構造化
            sections = re.split(r"\n\s*#{1,3}\s+", response_text)
            structured_data = {
                "analysis": {
                    "raw_text": response_text,
                    "sections": [s.strip() for s in sections if s.strip()],
                    "summary": response_text[:500],
                },
                "model_assessment": self._extract_assessment(response_text),
                "key_metrics": self._extract_metrics(response_text),
                "focus_area": self.default_focus_area,
                "analysis_type": "structured_extraction",
                "execution_time": time.time(),
            }

            self.log_json(
                "DEBUG",
                "Created structured data",
                {"data_structure": list(structured_data.keys())},
            )

            return structured_data

        except Exception as e:
            self.log_json(
                "ERROR",
                "Error in parse_gemini_response",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            return {
                "error": str(e),
                "raw_text": response_text[:1000],
                "focus_area": self.default_focus_area,
                "analysis_type": "error_response",
            }

    def extract_structured_data(self, text_response: str) -> dict:
        """
        テキスト応答から構造化データを抽出する
        """
        try:
            # デバッグログの追加
            self.log_json(
                "DEBUG",
                "Extracting structured data",
                {"text_preview": text_response[:200]},
            )

            # デフォルトの応答構造
            structured_data = {
                "model_assessment": 0,
                "key_metrics": [],
                "risk_factors": [],
                "optimization_strategies": [],
                "innovation_proposals": [],
                "market_fit_analysis": {
                    "score": 0,
                    "justification": "Unable to parse response",
                },
                "raw_response": text_response,
            }

            # テキストを行ごとに分析
            current_section = None
            lines = text_response.split("\n")

            for i, line in enumerate(lines):
                try:
                    line = line.strip()
                    if not line:
                        continue

                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().lower().replace(" ", "_")
                        value = value.strip()

                        # キーが既知のフィールドの場合
                        if key in structured_data:
                            if isinstance(structured_data[key], list):
                                structured_data[key].append(
                                    {"description": value, "line_number": i + 1}
                                )
                            else:
                                structured_data[key] = value
                        # セクションヘッダーの場合
                        elif key in [
                            "key_metrics",
                            "risk_factors",
                            "optimization_strategies",
                            "innovation_proposals",
                        ]:
                            current_section = key
                            self.log_json(
                                "DEBUG", f"Found section: {key}", {"line_number": i + 1}
                            )
                        # 現在のセクションに属する項目の場合
                        elif current_section:
                            structured_data[current_section].append(
                                {"description": line, "line_number": i + 1}
                            )

                except Exception as line_error:
                    self.log_json(
                        "WARNING",
                        f"Error processing line {i + 1}",
                        {"error": str(line_error), "line": line},
                    )
                    continue

            # データの検証
            if not any(
                structured_data[key]
                for key in [
                    "key_metrics",
                    "risk_factors",
                    "optimization_strategies",
                    "innovation_proposals",
                ]
            ):
                self.log_json(
                    "WARNING",
                    "No structured data extracted",
                    {"text_length": len(text_response)},
                )

            # market_fit_analysisのスコア正規化
            try:
                if isinstance(structured_data["market_fit_analysis"], dict):
                    score = float(structured_data["market_fit_analysis"]["score"])
                    structured_data["market_fit_analysis"]["score"] = min(
                        max(score, 0), 100
                    )
            except (ValueError, KeyError) as e:
                self.log_json(
                    "WARNING", "Error normalizing market fit score", {"error": str(e)}
                )

            return structured_data

        except Exception as e:
            self.log_json(
                "ERROR",
                "Error extracting structured data",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            return {
                "error": str(e),
                "raw_response": text_response,
                "timestamp": datetime.now().isoformat(),
            }

    def generate_analysis_prompt(self, input_data, unique_focus):
        prompt = f"""Analyze the following token economics data and provide data-driven insights and visualization suggestions. {unique_focus} Your response should be in a structured JSON format for efficient processing by other AI agents.

Input data:
{json.dumps(input_data, indent=2)}

Respond with a JSON object containing the following keys:
1. model_assessment: A numerical score (0-100) representing the overall quality of the token economics model.
2. key_metrics: An array of important metrics extracted from the input data, each with a 'metric_name', 'value', and 'description'.
3. risk_factors: An array of potential risks identified in the model, each with a 'risk_type', 'description', and 'mitigation_strategy'.
4. optimization_strategies: An array of data-driven strategies to improve the model, each with a 'strategy', 'impact_score' (0-100), and 'implementation_steps'.
5. innovation_proposals: An array of innovative features or approaches, each with a 'proposal', 'feasibility_score' (0-100), and 'potential_impact'.
6. market_fit_analysis: An object with 'score' (0-100) representing how well the model fits current market trends, 'justification' providing a brief explanation, and 'improvement_suggestions'.

Ensure your response is comprehensive, focusing on critical analysis, ethical implications, and actionable recommendations for improving the token economics model's robustness and compliance. Provide detailed descriptions for all fields, avoiding generic responses."""

        return prompt

    def process_gemini_response(self, gemini_response, unique_focus):
        """Geminiのレスポンスを処理し、標準化された形式に変換する"""
        try:
            # デバッグログの追加
            self.log_json(
                "DEBUG",
                "Processing Gemini response",
                {
                    "response_type": type(gemini_response).__name__,
                    "unique_focus": unique_focus,
                },
            )

            # gemini_responseがNoneの場合のチェック
            if gemini_response is None:
                self.log_json("ERROR", "Received None response from Gemini API")
                gemini_response = {}

            # レスポンス構造の作成（focus_areaを最優先で設定）
            base_content = self.set_default_values(gemini_response)
            base_content["focus_area"] = unique_focus  # content内にfocus_area追加

            standardized_response = {
                "focus_area": unique_focus,  # トップレベルに設定
                "content": {
                    **base_content,
                    "analysis_type": "token_economics_analysis",  # 追加
                    "focus_area": unique_focus,  # content内でも明示的に設定
                },
                "metadata": {
                    "agent_name": self.agent_name,
                    "focus_area": unique_focus,  # メタデータ内でも設定
                    "timestamp": datetime.now().isoformat(),
                    "standardization_version": "1.0",
                    "processing_status": "completed",  # 追加
                },
            }

            # コンテンツの標準化
            standardized_response["content"] = self.standardize_data(
                standardized_response["content"]
            )

            # より詳細なレスポンス構造の検証ログ
            self.log_json(
                "DEBUG",
                "Response structure verification",
                {
                    "top_level_focus": standardized_response.get("focus_area"),
                    "content_focus": standardized_response["content"].get("focus_area"),
                    "metadata_focus": standardized_response["metadata"].get(
                        "focus_area"
                    ),
                    "has_all_focus_areas": all(
                        [
                            standardized_response.get("focus_area"),
                            standardized_response["content"].get("focus_area"),
                            standardized_response["metadata"].get("focus_area"),
                        ]
                    ),
                },
            )

            return standardized_response

        except Exception as e:
            self.log_json(
                "ERROR",
                "Error in process_gemini_response",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            # エラー時のフォールバック（focus_areaを必ず含める）
            return {
                "focus_area": unique_focus,
                "error": str(e),
                "metadata": {
                    "agent_name": self.agent_name,
                    "focus_area": unique_focus,
                    "error": True,
                    "timestamp": datetime.now().isoformat(),
                    "processing_status": "error",  # 追加
                },
            }

    async def handle_error(
        self, e: Exception, receiver: str, start_time: float, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """エラーハンドリング処理"""
        error_message = f"Error in Gemini processing: {str(e)}"
        self.log_json("ERROR", error_message, {"traceback": traceback.format_exc()})
        execution_time = time.time() - start_time
        partial_results = self.extract_partial_results(input_data)

        # CommunicationProtocolのインスタンスメソッドとして呼び出し
        return await self.communication_protocol.create_message(
            sender=self.agent_name,
            receiver=receiver,
            message_type="error",
            content={
                "error_code": "GEMINI_PROCESSING_ERROR",
                "error_message": error_message,
                "execution_time": execution_time,
                "partial_results": partial_results,
                "focus_area": self.default_focus_area,
            },
        )

    def extract_partial_results(self, data):
        if isinstance(data, dict):
            return {
                k: self.extract_partial_results(v)
                for k, v in data.items()
                if not isinstance(v, (dict, list))
            }
        elif isinstance(data, list):
            return [
                self.extract_partial_results(item)
                for item in data
                if not isinstance(item, (dict, list))
            ]
        else:
            return data

    def _extract_metrics(self, text: str) -> List[Dict[str, Any]]:
        """テキストからメトリクスを抽出"""
        metrics = []
        patterns = {
            "percentage": r"(\d+(?:\.\d+)?)\s*%",
            "token_amount": r"(\d+(?:\.\d+)?)\s*tokens?",
            "price": r"\$\s*(\d+(?:\.\d+)?)",
            "time_period": r"(\d+)\s*(day|week|month|year)s?",
        }

        for metric_type, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                context = text[
                    max(0, match.start() - 50) : min(len(text), match.end() + 50)
                ]
                metrics.append(
                    {
                        "metric_type": metric_type,
                        "value": match.group(1),
                        "unit": (
                            "%"
                            if metric_type == "percentage"
                            else (
                                "tokens"
                                if metric_type == "token_amount"
                                else "$" if metric_type == "price" else "time"
                            )
                        ),
                        "context": context.strip(),
                    }
                )

        return metrics

    def _extract_assessment(self, text: str) -> int:
        """テキストからモデル評価スコアを抽出"""
        try:
            # 数値を探す（0-100のスコア）
            matches = re.findall(r"(\d+)(?=\s*\/\s*100|\s*%)", text)
            if matches:
                score = int(matches[0])
                return min(max(score, 0), 100)  # 0-100の範囲に制限

            # 数値のみの場合
            matches = re.findall(r"\b(\d+)\b", text)
            if matches:
                score = int(matches[0])
                return min(max(score, 0), 100)

            return 0
        except Exception as e:
            self.log_json(
                "ERROR",
                "Error extracting assessment score",
                {"error": str(e), "text": text[:100]},
            )
            return

    def _get_unit(self, metric_type: str) -> str:
        """メトリクスタイプに基づいて単位を返す"""
        units = {
            "percentage": "%",
            "token_amount": "tokens",
            "price": "$",
            "time_period": "time",
        }
        return units.get(metric_type, "unknown")

    @classmethod
    async def main(cls):
        """メインの実行メソッド"""
        try:
            agent = cls()
            comm_protocol = CommunicationProtocol()

            test_input = {
                "token_supply": 1000000,
                "initial_price": 0.1,
                "inflation_rate": 0.05,
                "staking_reward": 0.08,
                "burn_mechanism": "2% of transaction fees",
                "vesting_schedule": "10% unlock per quarter",
            }

            # メッセージ作成
            test_message = await comm_protocol.create_message(
                sender="System",
                receiver="gemini",
                message_type="token_economics_data",
                content=test_input,
            )

            # クエリ実行
            result = await agent.query_gemini(test_message)

            print(json.dumps(result, indent=2))
            return result

        except Exception as e:
            agent.log_json(
                "ERROR",
                "Error in main",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            raise


if __name__ == "__main__":
    asyncio.run(GeminiAgent.main())
