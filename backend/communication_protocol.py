import asyncio
import json
import logging
import traceback
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CommunicationProtocol:
    def __init__(self):
        """初期化"""
        self.agent_functions = {}
        self.default_focus_area = "token_economics_analysis"
        self.logger = logging.getLogger(__name__)

    def log_json(self, level: str, message: str, extra: Dict = None) -> None:
        """JSONフォーマットでログを出力する"""
        try:
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

            # ログレベルの検証
            valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            log_level = level.upper()
            if log_level not in valid_levels:
                log_level = "ERROR"
                log_data["original_level"] = level

            self.logger.log(
                getattr(logging, log_level), json.dumps(log_data, ensure_ascii=False)
            )

        except Exception as e:
            self.logger.error(f"Logging error: {str(e)}")

    def register_agent(self, agent_name: str, agent_function: Callable):
        """エージェントの登録"""
        agent_name = agent_name.lower()
        self.agent_functions[agent_name] = agent_function
        logger.debug(f"Registered agent: {agent_name}")
        logger.debug(f"Current registered agents: {self.get_registered_agents()}")
        logger.info(
            f"Registered agent: {agent_name}, Total agents: {len(self.agent_functions)}"
        )

    def get_registered_agents(self):
        return list(self.agent_functions.keys())

    def validate_agent_name(self, agent_name: str) -> bool:
        """エージェント名が有効かどうかを検証する"""
        normalized_name = agent_name.lower()
        is_valid = normalized_name in self.agent_functions
        logger.debug(
            f"Validating agent name: {agent_name} (normalized: {normalized_name}) - Valid: {is_valid}"
        )
        return is_valid

    async def create_message(
        self, sender: str, receiver: str, message_type: str, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """メッセージの作成（非同期バージョン）"""
        try:
            # デバッグログの追加
            self.log_json(
                "DEBUG",
                "Creating message",
                {
                    "sender": sender,
                    "receiver": receiver,
                    "message_type": message_type,
                    "content_structure": str(content)[:200],
                },
            )

            sender = sender.lower()
            receiver = receiver.lower()

            # メッセージタイプに基づくデフォルトのfocus_area設定
            default_focus_mapping = {
                "token_economics_data": "Token Economics Analysis",
                "market_analysis": "Market Analysis",
                "technical_analysis": "Technical Analysis",
                "response": content.get("focus_area", self.default_focus_area),
            }

            # focus_areaの決定
            focus_area = content.get("focus_area") or default_focus_mapping.get(
                message_type, self.default_focus_area
            )

            # メッセージ構造の作成
            message = {
                "sender": sender,
                "receiver": receiver,
                "message_type": message_type,
                "content": {
                    **content,
                    "focus_area": focus_area,  # content内にfocus_areaを設定
                    "analysis_type": message_type,
                },
                "focus_area": focus_area,  # トップレベルに設定
                "metadata": {
                    "agent_name": sender,
                    "focus_area": focus_area,  # メタデータ内にも設定
                    "timestamp": datetime.now().isoformat(),
                    "standardization_version": "1.0",
                    "message_type": message_type,
                },
                "timestamp": datetime.now().isoformat(),
            }

            # 検証ログ
            self.log_json(
                "DEBUG",
                "Message structure verification",
                {
                    "has_focus_area": "focus_area" in message,
                    "focus_area": focus_area,
                    "message_type": message_type,
                    "message_structure": list(message.keys()),
                    "content_keys": list(message["content"].keys()),
                },
            )

            return message

        except Exception as e:
            self.log_json(
                "ERROR",
                "Error creating message",
                {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "sender": sender,
                    "receiver": receiver,
                },
            )
            return {
                "sender": "system",
                "receiver": receiver,
                "message_type": "error",
                "content": {
                    "error": str(e),
                    "original_sender": sender,
                    "focus_area": self.default_focus_area,
                },
                "focus_area": self.default_focus_area,
                "metadata": {
                    "agent_name": "system",
                    "error": True,
                    "timestamp": datetime.now().isoformat(),
                    "focus_area": self.default_focus_area,
                },
                "timestamp": datetime.now().isoformat(),
            }

    # create_message_asyncのエイリアスを追加
    create_message_async = create_message

    async def route_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """メッセージのルーティング"""
        try:
            if "action" in message and message["action"] == "Finish":
                return await self.create_message(
                    sender="System",
                    receiver=message.get("sender", "unknown"),
                    message_type="finish",
                    content={"status": "completed"},
                )

            receiver = message["receiver"].lower()
            logger.debug(f"Attempting to route message to: {receiver}")

            if receiver not in self.agent_functions:
                error_message = f"Unknown receiver: {receiver}. Available agents: {self.get_registered_agents()}"
                logger.error(error_message)
                return await self.create_message(
                    sender="System",
                    receiver=message["sender"],
                    message_type="error",
                    content={"error": error_message},
                )

            agent_function = self.agent_functions[receiver]
            response_content = await agent_function(message)
            standardized_response = self.standardize_response(
                response_content, agent_name=receiver
            )

            return await self.create_message(
                sender=receiver,
                receiver=message["sender"],
                message_type="response",
                content=standardized_response,
            )

        except Exception as e:
            error_message = f"Error processing message for {receiver}: {str(e)}"
            logger.error(error_message)
            return await self.create_message(
                sender="System",
                receiver=message["sender"],
                message_type="error",
                content={
                    "error_code": "PROCESSING_ERROR",
                    "error_message": error_message,
                    "partial_results": None,
                },
            )

    def _extract_focus_area(self, content: Dict[str, Any]) -> str:
        """focus_areaの抽出とデフォルト値の設定"""
        try:
            # デバッグログの追加
            self.log_json(
                "DEBUG",
                "Extracting focus area",
                {
                    "content_type": type(content).__name__,
                    "content_keys": (
                        list(content.keys()) if isinstance(content, dict) else None
                    ),
                },
            )

            if not isinstance(content, dict):
                return self.default_focus_area

            # メッセージタイプに基づくデフォルトのfocus_area設定
            message_type = content.get("message_type", "")
            default_focus_mapping = {
                "token_economics_data": "Token Economics Analysis",
                "market_analysis": "Market Analysis",
                "technical_analysis": "Technical Analysis",
                "risk_assessment": "Risk Assessment",
            }

            # 階層的な検索
            focus_area = None

            # 1. previous_responsesからの検索
            if "previous_responses" in content:
                for agent, response in content["previous_responses"].items():
                    if isinstance(response, dict):
                        if "original_response" in response:
                            orig_resp = response["original_response"]
                            if (
                                isinstance(orig_resp, dict)
                                and "focus_area" in orig_resp
                            ):
                                focus_area = orig_resp["focus_area"]
                                self.log_json(
                                    "DEBUG",
                                    "Found focus_area in previous_responses",
                                    {"agent": agent, "focus_area": focus_area},
                                )
                                return focus_area

            # 2. 直接のfocus_area
            if "focus_area" in content and content["focus_area"]:
                return content["focus_area"]

            # 3. content内のfocus_area
            if "content" in content and isinstance(content["content"], dict):
                if "focus_area" in content["content"]:
                    return content["content"]["focus_area"]

            # 4. メタデータ内のfocus_area
            metadata = content.get("metadata", {})
            if isinstance(metadata, dict):
                if "focus_area" in metadata and metadata["focus_area"]:
                    return metadata["focus_area"]

            # 5. メッセージタイプからのデフォルト値
            if message_type in default_focus_mapping:
                return default_focus_mapping[message_type]

            # 警告ログの詳細化
            self.log_json(
                "WARNING",
                "focus_area not found in response or content, using default value",
                {
                    "content_keys": list(content.keys()),
                    "has_previous_responses": "previous_responses" in content,
                    "has_content": "content" in content,
                    "has_metadata": "metadata" in content,
                    "message_type": message_type,
                    "available_agents": list(
                        content.get("previous_responses", {}).keys()
                    ),
                },
            )

            return self.default_focus_area

        except Exception as e:
            self.log_json(
                "ERROR",
                "Error extracting focus_area",
                {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "content_type": type(content).__name__,
                },
            )
            return self.default_focus_area

    @staticmethod
    async def validate_message(message: Dict[str, Any]) -> bool:
        required_fields = ["sender", "receiver", "message_type", "content", "timestamp"]
        if not all(field in message for field in required_fields):
            logger.error(f"Invalid message format: {message}")
            return False
        return True

    # "Finish" アクションを処理するメソッドを追加
    def handle_finish_action(self, message: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Handling Finish action")
        return {
            "status": "completed",
            "message": "Analysis finished successfully",
            "sender": message.get("sender", "Unknown"),
            "timestamp": datetime.now().isoformat(),
        }

    def standardize_response(
        self, response: Dict[str, Any], agent_name: str = None
    ) -> Dict[str, Any]:
        """レスポンスを標準化する"""
        try:
            # デバッグログの追加
            self.log_json(
                "DEBUG",
                "Standardizing response",
                {
                    "response_type": type(response).__name__,
                    "agent_name": agent_name,
                    "response_keys": (
                        list(response.keys()) if isinstance(response, dict) else None
                    ),
                },
            )

            content = response.get("content", {})
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    self.log_json(
                        "WARNING", f"Failed to parse content as JSON: {content[:200]}"
                    )
                    content = {}

            # focus_areaの階層的な検索
            focus_area = (
                response.get("focus_area")
                or content.get("focus_area")
                or response.get("metadata", {}).get("focus_area")
                or (
                    response.get("original_response", {}).get("focus_area")
                    if isinstance(response.get("original_response"), dict)
                    else None
                )
                or "General token economics analysis"
            )

            # 標準化されたレスポンスの構築
            standardized = {
                "focus_area": focus_area,  # トップレベルに設定
                "original_response": response,
                "content": {**content, "focus_area": focus_area},  # content内にも設定
                "metadata": {
                    "standardized_at": datetime.now().isoformat(),
                    "standardization_version": "1.0",
                    "agent_name": agent_name or response.get("sender", "Unknown"),
                    "focus_area": focus_area,  # metadata内にも設定
                },
            }

            # レスポンス構造の検証ログ
            self.log_json(
                "DEBUG",
                "Standardized response structure",
                {
                    "has_focus_area": "focus_area" in standardized,
                    "focus_area_value": focus_area,
                    "response_structure": list(standardized.keys()),
                },
            )

            return standardized

        except Exception as e:
            self.log_json(
                "ERROR",
                "Error in standardize_response",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            return {
                "focus_area": "error_handling",
                "error": str(e),
                "metadata": {"error": True, "timestamp": datetime.now().isoformat()},
            }

    @staticmethod
    def standardize_field(key: str, value: Any, depth: int = 0) -> Any:
        if depth > 10:
            return str(value)

        try:
            if key == "model_assessment":
                return CommunicationProtocol.standardize_model_assessment(value)
            elif key == "key_metrics":
                return CommunicationProtocol.standardize_key_metrics(value, depth + 1)
            elif key == "risk_factors":
                return CommunicationProtocol.standardize_risk_factors(value, depth + 1)
            elif key == "optimization_strategies":
                return CommunicationProtocol.standardize_optimization_strategies(
                    value, depth + 1
                )
            elif key == "innovation_proposals":
                return CommunicationProtocol.standardize_innovation_proposals(
                    value, depth + 1
                )
            elif key == "market_fit_analysis":
                return CommunicationProtocol.standardize_market_fit_analysis(
                    value, depth + 1
                )
            elif key == "focus_area":
                return str(value) if value else None
            elif key == "execution_time":
                return float(value) if value else None
            elif isinstance(value, dict):
                return {
                    k: CommunicationProtocol.standardize_field(k, v, depth + 1)
                    for k, v in value.items()
                }
            elif isinstance(value, list):
                return [
                    CommunicationProtocol.standardize_field(key, item, depth + 1)
                    for item in value
                ]
            else:
                return value
        except Exception as e:
            logger.error(
                f"Error in standardize_field for {key}: {str(e)}", exc_info=True
            )
            return None

    @staticmethod
    def standardize_model_assessment(value: Any) -> int:
        if isinstance(value, dict) and "score" in value:
            return int(value["score"])
        elif isinstance(value, (int, float)):
            return int(value)
        else:
            logger.warning(f"Invalid model_assessment value: {value}")
            return None

    @staticmethod
    def standardize_key_metrics(
        metrics: List[Dict[str, Any]], depth: int
    ) -> List[Dict[str, Any]]:
        standardized_metrics = []
        for metric in metrics:
            if isinstance(metric, dict):
                standardized_metric = {
                    "metric_name": metric.get("metric_name"),
                    "value": metric.get("value"),
                    "description": metric.get("description", "No description provided"),
                }
                standardized_metrics.append(standardized_metric)
            else:
                logger.warning(f"Invalid key_metric: {metric}")
        return standardized_metrics

    @staticmethod
    def standardize_risk_factors(
        risks: List[Dict[str, Any]], depth: int
    ) -> List[Dict[str, Any]]:
        standardized_risks = []
        for risk in risks:
            if isinstance(risk, dict):
                standardized_risk = {
                    "risk_type": risk.get("risk_type"),
                    "description": risk.get("description", "No description provided"),
                    "mitigation_strategy": risk.get(
                        "mitigation_strategy", "No mitigation strategy provided"
                    ),
                }
                standardized_risks.append(standardized_risk)
            else:
                logger.warning(f"Invalid risk_factor: {risk}")
        return standardized_risks

    @staticmethod
    def standardize_optimization_strategies(
        strategies: List[Dict[str, Any]], depth: int
    ) -> List[Dict[str, Any]]:
        standardized_strategies = []
        for strategy in strategies:
            if isinstance(strategy, dict):
                standardized_strategy = {
                    "strategy": strategy.get("strategy"),
                    "description": strategy.get(
                        "description", "No description provided"
                    ),
                    "impact_score": strategy.get("impact_score", 0),
                }
                standardized_strategies.append(standardized_strategy)
            else:
                logger.warning(f"Invalid optimization_strategy: {strategy}")
        return standardized_strategies

    @staticmethod
    def standardize_innovation_proposals(
        proposals: List[Dict[str, Any]], depth: int
    ) -> List[Dict[str, Any]]:
        standardized_proposals = []
        for proposal in proposals:
            if isinstance(proposal, dict):
                standardized_proposal = {
                    "proposal": proposal.get("proposal"),
                    "description": proposal.get(
                        "description", "No description provided"
                    ),
                    "feasibility_score": proposal.get("feasibility_score", 0),
                }
                standardized_proposals.append(standardized_proposal)
            else:
                logger.warning(f"Invalid innovation_proposal: {proposal}")
        return standardized_proposals

    @staticmethod
    def standardize_market_fit_analysis(analysis: Any, depth: int) -> Dict[str, Any]:
        if isinstance(analysis, dict):
            return {
                "score": analysis.get("score"),
                "justification": analysis.get(
                    "justification", "No justification provided"
                ),
            }
        elif isinstance(analysis, (int, float)):
            return {
                "score": int(analysis),
                "justification": "No justification provided",
            }
        else:
            logger.warning(f"Invalid market_fit_analysis: {analysis}")
            return {"score": None, "justification": "Invalid market fit analysis"}

    @staticmethod
    def ensure_consistency(data: Dict[str, Any]) -> Dict[str, Any]:
        if "model_assessment" in data and data["model_assessment"] is not None:
            data["model_assessment"] = max(0, min(100, data["model_assessment"]))

        if "market_fit_analysis" in data and isinstance(
            data["market_fit_analysis"], dict
        ):
            if (
                "score" in data["market_fit_analysis"]
                and data["market_fit_analysis"]["score"] is not None
            ):
                data["market_fit_analysis"]["score"] = max(
                    0, min(100, data["market_fit_analysis"]["score"])
                )

        for strategy in data.get("optimization_strategies", []):
            if "impact_score" in strategy and strategy["impact_score"] is not None:
                strategy["impact_score"] = max(0, min(100, strategy["impact_score"]))

        for proposal in data.get("innovation_proposals", []):
            if (
                "feasibility_score" in proposal
                and proposal["feasibility_score"] is not None
            ):
                proposal["feasibility_score"] = max(
                    0, min(100, proposal["feasibility_score"])
                )

        return data

    @staticmethod
    def remove_duplicates(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique_items = []
        for item in items:
            if not any(
                CommunicationProtocol.is_similar(item, existing)
                for existing in unique_items
            ):
                unique_items.append(item)
        return unique_items

    @staticmethod
    def is_similar(
        item1: Dict[str, Any], item2: Dict[str, Any], threshold: float = 0.8
    ) -> bool:
        text1 = json.dumps(item1, sort_keys=True)
        text2 = json.dumps(item2, sort_keys=True)
        return SequenceMatcher(None, text1, text2).ratio() > threshold

    @staticmethod
    def remove_circular_references(data, seen=None):
        if seen is None:
            seen = set()

        if isinstance(data, dict):
            new_dict = {}
            for key, value in data.items():
                if id(value) in seen:
                    new_dict[key] = "Circular reference detected"
                else:
                    seen.add(id(value))
                    new_dict[key] = CommunicationProtocol.remove_circular_references(
                        value, seen
                    )
                    seen.remove(id(value))
            return new_dict
        elif isinstance(data, list):
            new_list = []
            for item in data:
                if id(item) in seen:
                    new_list.append("Circular reference detected")
                else:
                    seen.add(id(item))
                    new_list.append(
                        CommunicationProtocol.remove_circular_references(item, seen)
                    )
                    seen.remove(id(item))
            return new_list
        else:
            return data

    @staticmethod
    def json_serialize(data):
        return json.dumps(
            CommunicationProtocol.remove_circular_references(data),
            default=str,
            indent=2,
        )

    def get_agent_function(self, agent_name: str) -> Callable:
        if agent_name not in self.agent_functions:
            raise ValueError(f"Agent '{agent_name}' is not registered")
        return self.agent_functions[agent_name]
