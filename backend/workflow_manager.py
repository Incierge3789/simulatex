import asyncio
import json
import yaml
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging
from difflib import SequenceMatcher
from collections import defaultdict
from functools import lru_cache
import hashlib

# エージェントのインポートを修正
from agents.gpt4_agent import GPT4Agent
from agents.claude_agent import ClaudeAgent
from agents.gemini_agent import GeminiAgent
from agents.analyst_ai import AnalystAI
from communication_protocol import CommunicationProtocol
from data_manager import DataManager  # DataManagerをインポート

# ロガーの設定
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class WorkflowManager:
    def __init__(self):
        self.workflows = self.load_workflows()
        self.communication_protocol = CommunicationProtocol()
        self.registered_agents = set()
        self.register_agents()
        self.agent_weights = self.load_agent_weights()
        self.cache = {}
        # # DataManagerの初期化を遅延
        self.data_manager = None

    async def ensure_initialized(self):
        """DataManagerの初期化を確認"""
        if self.data_manager is None:
            try:
                self.data_manager = DataManager()
                await self.data_manager.ensure_initialized()
                logger.info("DataManager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize DataManager: {str(e)}")
                raise

    # 5. 動的なワークフロー管理
    def load_workflows(self):
        # 外部設定ファイルからワークフローの定義を読み込む
        with open('workflows.yaml', 'r') as file:
            return yaml.safe_load(file)

    # 7. 重み付け評価システム
    def load_agent_weights(self):
        # 外部設定ファイルからエージェントの重みを読み込む
        with open('agent_weights.yaml', 'r') as file:
            return yaml.safe_load(file)

    def register_agents(self):
        # エージェントのインスタンスを作成
        agents = {
            'analyst_ai': AnalystAI().query_analyst_ai,
            'claude': ClaudeAgent().query_claude,
            'gemini': GeminiAgent().query_gemini,
            'gpt4': GPT4Agent().query_gpt4
        }

        for agent_name, query_func in agents.items():
            self.register_agent(agent_name, query_func)

        logger.info(f"Registered agents: {list(self.registered_agents)}")

    def register_agent(self, agent_name: str, query_func):
        """
        エージェントを登録する関数。重複登録を防ぐ。
        """
        if agent_name not in self.registered_agents:
            self.communication_protocol.register_agent(agent_name, lambda message, agent=agent_name: query_func(message))
            self.registered_agents.add(agent_name)
            logger.info(f"Registered agent: {agent_name}")
        else:
            logger.info(f"Agent {agent_name} already registered. Skipping.")

    def get_agent_function(self, agent_name: str):
        """
        エージェント名に応じた関数を返す
        """
        agent_functions = {
            'analyst_ai': AnalystAI().query_analyst_ai,
            'claude': ClaudeAgent().query_claude,
            'gemini': GeminiAgent().query_gemini,
            'gpt4': GPT4Agent().query_gpt4
        }
        return agent_functions.get(agent_name)

    async def execute_workflow(self, workflow_name: str, initial_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ワークフローを実行する"""
        logger.info(f"Starting workflow: {workflow_name}")
        logger.debug(f"Initial data: {initial_data}")

        if workflow_name not in self.workflows:
            logger.error(f"Workflow '{workflow_name}' not found")
            raise ValueError(f"Workflow '{workflow_name}' not found")

        workflow = self.workflows[workflow_name]
        results = []
        current_data = initial_data.copy()

        try:
            # 現在のイベントループを使用
            loop = asyncio.get_event_loop()
            for step in workflow:
                try:
                    agent = step["agent"]
                    next_step = step["next"]
                    receiver = next_step if next_step else "WorkflowManager"

                    logger.info(f"Executing step: {step['name']} with agent: {agent}")
                    logger.debug(f"Input data for {agent}: {current_data}")

                    # データの検証と補完
                    validated_data = await self.validate_and_enrich_data(current_data)

                    # メッセージ作成（同じループを使用）
                    message = await self.communication_protocol.create_message(
                        sender="WorkflowManager",
                        receiver=agent,
                        message_type="analysis_request",
                        content=validated_data
                    )

                    # ステップ実行
                    result = await self.execute_step(step, message, receiver)
                    results.append(result)

                    if result["message_type"] != "error":
                        current_data = self.merge_data(current_data, result["content"])
                        await self.share_data_between_agents(agent, result["content"])
                    else:
                        logger.warning(f"Error in step {step['name']}: {result['content'].get('error', 'Unknown error')}")

                except Exception as e:
                    logger.error(f"Error in step {step['name']}: {str(e)}", exc_info=True)
                    results.append({
                        "message_type": "error",
                        "content": {"error": str(e), "step": step['name']}
                    })

            logger.info(f"Workflow {workflow_name} completed")
            return results

        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}", exc_info=True)
            raise

    async def validate_and_enrich_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """データの検証と補完"""
        try:
            # # 初期化の確認
            await self.ensure_initialized()
            
            required_fields = [
                'token_symbol',
                'total_supply',
                'circulating_supply',
                'inflation_rate',
                'staking_reward'
            ]
            
            # # データの取得と基本的な検証
            for field in required_fields:
                if field not in data or data[field] is None or data[field] == "Unknown":
                    try:
                        # # 非同期処理を直接実行
                        fetched_data = await self.data_manager.get_token_economics_data(
                            data.get('token_symbol', 'Unknown')
                        )
                        
                        if fetched_data and field in fetched_data:
                            data[field] = fetched_data[field]
                        else:
                            data[field] = "Unknown"
                    except Exception as e:
                        logger.error(f"Error fetching data for {field}: {str(e)}")
                        data[field] = "Unknown"

            # # インフレ率の検証と正規化
            if data['inflation_rate'] != "Unknown":
                try:
                    inflation_rate = self._normalize_inflation_rate(
                        data['inflation_rate']
                    )
                    if inflation_rate is not None:
                        data['inflation_rate'] = inflation_rate
                        if inflation_rate > 0.2:
                            logger.warning(
                                f"Unusually high inflation rate: {inflation_rate * 100}%"
                            )
                            data['inflation_warning'] = "High inflation rate detected"
                except Exception as e:
                    logger.error(f"Error processing inflation rate: {str(e)}")
                    data['inflation_rate'] = "Unknown"

            return data
                
        except Exception as e:
            logger.error(f"Error in validate_and_enrich_data: {str(e)}")
            return data

    def _normalize_inflation_rate(self, rate_value: Any) -> Optional[float]:
        """インフレ率の正規化"""
        try:
            # # 文字列の場合
            if isinstance(rate_value, str):
                rate_str = rate_value.strip('%')
                rate = float(rate_str)
            else:
                # # 数値の場合
                rate = float(rate_value)

            # # パーセント表記から小数点表記への変換
            if rate > 1:  # 1より大きい場合はパーセント表記と判断
                rate = rate / 100

            # # 値の範囲チェック（0-100%）
            if 0 <= rate <= 1:
                return rate
            else:
                logger.error(f"Inflation rate out of valid range: {rate}")
                return None

        except (ValueError, AttributeError) as e:
            logger.error(f"Invalid inflation rate value: {rate_value}")
            return None


    async def share_data_between_agents(self, source_agent: str, data: Dict[str, Any]):
        """
        エージェント間でデータを共有する
        """
        for agent in self.communication_protocol.get_registered_agents():
            if agent != source_agent:
                message = await self.communication_protocol.create_message(
                    sender=source_agent,
                    receiver=agent,
                    message_type="shared_data",
                    content=data
                )
                await self.communication_protocol.route_message(message)



    # 新しく追加された動的ワークフロー調整機能
    async def adjust_workflow(self, workflow_name: str, results: List[Dict[str, Any]]) -> None:
        logger.info(f"Adjusting workflow: {workflow_name}")
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            logger.warning(f"Workflow '{workflow_name}' not found for adjustment")
            return

        # 結果に基づいてワークフローを動的に調整
        adjusted_workflow = []
        for step in workflow:
            adjusted_workflow.append(step)
            
            # 特定の条件に基づいて追加のステップを挿入
            if step["agent"] == "Analyst AI" and self.needs_additional_analysis(results):
                adjusted_workflow.append({
                    "name": "Additional Market Analysis",
                    "agent": "gpt4",
                    "next": step["next"]
                })
        
        self.workflows[workflow_name] = adjusted_workflow
        logger.info(f"Workflow '{workflow_name}' adjusted")

    def needs_additional_analysis(self, results: List[Dict[str, Any]]) -> bool:
        # 結果を分析して追加の分析が必要かどうかを判断
        for result in results:
            if result.get("message_type") == "market_analysis":
                content = result.get("content", {})
                if content.get("market_fit_analysis", {}).get("score", 0) < 50:
                    return True
        return False

    # execute_step関数も改善し、より詳細なエラーハンドリングを実装
    async def execute_step(self, step, message, receiver):
        try:
            logger.debug(f"Executing step: {step['name']} with message: {message}")
            result = await self.communication_protocol.route_message(message)
            logger.debug(f"Received result from {step['name']}: {result}")

            if result["message_type"] == "error":
                error_message = result['content'].get('error', 'Unknown error occurred')
                logger.error(f"Error in {step['name']}: {error_message}")
                return result

            return result
        except Exception as e:
            error_message = f"Error in {step['name']}: {str(e)}"
            logger.exception(error_message)
            return {
                "message_type": "error",
                "content": {
                    "error_code": "WORKFLOW_EXECUTION_ERROR",
                    "error_message": error_message,
                    "step": step['name'],
                    "partial_data": message['content']
                }
            }

    def merge_data(self, current_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        現在のデータと新しいデータをマージする関数
        """
        merged_data = current_data.copy()
        for key, value in new_data.items():
            if key in merged_data and isinstance(merged_data[key], dict) and isinstance(value, dict):
                merged_data[key] = self.merge_data(merged_data[key], value)
            else:
                merged_data[key] = value
        return merged_data

    def validate_token_data(self, data: Dict[str, Any]) -> bool:
        required_fields = {
            'token_symbol': 'Unknown',
            'total_supply': '0',
            'circulating_supply': '0',
            'inflation_rate': '5%',
            'staking_reward': '0%'
        }
        
        for field, default_value in required_fields.items():
            if field not in data or not data[field]:
                data[field] = default_value
                logger.warning(f"Missing required field: {field}")
        
        return True

    async def run_token_economics_analysis(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Starting token economics analysis")
        try:
            results = await self.execute_workflow("token_economics_analysis", token_data)
            final_result = {
                "workflow_name": "token_economics_analysis",
                "timestamp": datetime.now().isoformat(),
                "steps": results,
                "summary": self.generate_summary(results)
            }
            final_result = self.preprocess_for_storage(final_result)
            logger.info("Token economics analysis completed")
            return final_result
        except Exception as e:
            logger.error(f"Error in token economics analysis: {str(e)}", exc_info=True)
            return {
                "error": "An unexpected error occurred during the token economics analysis",
                "error_details": str(e),
                "partial_results": results if 'results' in locals() else None
            }

    # キャッシュ機能の強化
    @lru_cache(maxsize=100)
    def cache_result(self, input_data: str) -> Dict[str, Any]:
        # 入力データのハッシュを計算
        input_hash = hashlib.md5(input_data.encode()).hexdigest()
        
        # キャッシュから結果を取得
        cached_result = self.cache.get(input_hash)
        if cached_result:
            logger.info(f"Cache hit for input hash: {input_hash}")
            return cached_result

        # キャッシュにない場合は計算を実行
        result = self.perform_calculation(input_data)
        
        # 結果をキャッシュに保存
        self.cache[input_hash] = result
        logger.info(f"Cached result for input hash: {input_hash}")
        
        return result

    def perform_calculation(self, input_data: str) -> Dict[str, Any]:
        # 実際の計算ロジックをここに実装
        # この例では、簡単な計算を行います
        data = json.loads(input_data)
        result = {
            "total_tokens": sum(item.get("token_supply", 0) for item in data),
            "average_price": sum(item.get("initial_price", 0) for item in data) / len(data) if data else 0,
        }
        return result



    # generate_summary関数を改善し、エラーや部分的な結果をより適切に処理
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("Generating summary")
        try:
            cached_result = self.cached_calculation(json.dumps(results))
            logger.debug(f"Cached calculation result: {cached_result}")

            weighted_results = self.apply_weighted_evaluation(results)

            summary = {
                "overall_assessment": "",
                "key_insights": [],
                "risk_factors": [],
                "recommendations": [],
                "market_fit_analysis": {
                    "score": 0,
                    "justification": ""
                },
                "focus_areas": [],
                "agent_specific_analysis": {},
                "warnings": [],
                "errors": [],
                "cached_analysis": cached_result
            }

            valid_results = [r for r in results if r["message_type"] != "error"]
            error_results = [r for r in results if r["message_type"] == "error"]

            for result in valid_results:
                self.process_valid_result(result, summary)

            for error in error_results:
                summary["errors"].append(error["content"])

            if not valid_results:
                summary["overall_assessment"] = "Unable to generate assessment due to errors in all steps"
            else:
                summary["overall_assessment"] = self.calculate_overall_assessment(summary)

            summary["key_insights"] = self.extract_key_insights(summary["key_insights"])
            summary["risk_factors"], summary["recommendations"] = self.process_risks_and_recommendations(summary["risk_factors"], summary["recommendations"])
            summary["market_fit_analysis"] = self.integrate_market_fit_analysis(results)  # ここを修正
            summary["focus_areas"] = self.extract_focus_areas(summary["focus_areas"])

            summary["weighted_analysis"] = self.analyze_weighted_results(weighted_results)

            summary = self.ensure_data_consistency(summary)

            logger.debug(f"Generated summary: {summary}")
            return summary
        except Exception as e:
            logger.error(f"Error in generate_summary: {str(e)}", exc_info=True)
            return {
                "error": "An unexpected error occurred while generating the summary",
                "error_details": str(e),
                "partial_results": results
            }


    # 新しく追加された関数: 数値演算を安全に行う
    def safe_numeric_operation(self, value1: Any, value2: Any, operation: str) -> float:
        """
        数値演算を安全に行う関数
        """
        try:
            num1 = float(value1) if isinstance(value1, (int, float, str)) else 0
            num2 = float(value2) if isinstance(value2, (int, float, str)) else 0
            if operation == "subtract":
                return num1 - num2
            elif operation == "add":
                return num1 + num2
            elif operation == "multiply":
                return num1 * num2
            elif operation == "divide":
                return num1 / num2 if num2 != 0 else 0
            else:
                logger.error(f"Unsupported operation: {operation}")
                return 0
        except ValueError as e:
            logger.error(f"Error in numeric operation: {str(e)}")
            return 0

    # 新しい関数: 有効な結果を処理する
    def process_valid_result(self, result: Dict[str, Any], summary: Dict[str, Any]):
        content = result.get("content", {})
        if isinstance(content, dict):
            self.update_summary_fields(summary, content)
            agent_name = result.get("sender", "Unknown Agent")
            summary["agent_specific_analysis"][agent_name] = self.generate_agent_specific_analysis(content, agent_name)

    # 新しい関数: サマリーフィールドを更新する
    def update_summary_fields(self, summary: Dict[str, Any], content: Dict[str, Any]):
        for field in ["key_insights", "risk_factors", "recommendations", "focus_areas"]:
            if field in content:
                summary[field].extend(content[field])

        if "market_fit_analysis" in content:
            current_score = summary["market_fit_analysis"]["score"]
            new_score = content["market_fit_analysis"].get("score", 0)
            summary["market_fit_analysis"]["score"] = max(current_score, new_score)
            summary["market_fit_analysis"]["justification"] += " " + content["market_fit_analysis"].get("justification", "")


    # agent_specific_analysisの改善
    def generate_agent_specific_analysis(self, content: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        analysis = {
            "impact": "No impact analysis provided",
            "insights": "No additional insights provided"
        }

        try:
            # Impact analysis
            if "optimization_strategies" in content:
                strategies = content.get("optimization_strategies", [])
                if strategies:
                    top_strategy = max(strategies, key=lambda x: x.get("impact_score", 0))
                    analysis["impact"] = f"Highest impact strategy: {top_strategy.get('strategy', 'N/A')} (Impact score: {top_strategy.get('impact_score', 'N/A')})"

            # Additional insights
            insights = []
            if "innovation_proposals" in content:
                proposals = content.get("innovation_proposals", [])
                if proposals:
                    top_proposal = max(proposals, key=lambda x: x.get("feasibility_score", 0))
                    insights.append(f"Top innovation proposal: {top_proposal.get('proposal', 'N/A')} (Feasibility: {top_proposal.get('feasibility_score', 'N/A')})")
                    insights.append(f"Description: {top_proposal.get('description', 'No description provided')}")

            if "market_fit_analysis" in content:
                market_fit = content["market_fit_analysis"]
                if isinstance(market_fit, dict):
                    insights.append(f"Market fit score: {market_fit.get('score', 'N/A')}/100")
                    insights.append(f"Market fit justification: {market_fit.get('justification', 'No justification provided')}")

            if "focus_area" in content:
                insights.append(f"Focus area: {content.get('focus_area', 'No focus area specified')}")

            if insights:
                analysis["insights"] = " ".join(insights)

        except Exception as e:
            logger.error(f"Error in generate_agent_specific_analysis for {agent_name}: {str(e)}", exc_info=True)
            analysis["error"] = f"Error processing agent-specific analysis: {str(e)}"

        return analysis

    # データの一貫性チェックの強化
    def ensure_data_consistency(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Normalize scores
            if "overall_assessment" in summary:
                try:
                    assessment = summary["overall_assessment"]
                    if isinstance(assessment, str) and ":" in assessment:
                        score = float(assessment.split(":")[1].strip().split("/")[0])
                        summary["overall_assessment"] = f"Model Assessment: {self.normalize_scores([score])[0]:.2f}/100"
                    elif isinstance(assessment, (int, float)):
                        summary["overall_assessment"] = f"Model Assessment: {self.normalize_scores([assessment])[0]:.2f}/100"
                    else:
                        summary["overall_assessment"] = "Model Assessment: N/A"
                except (IndexError, ValueError) as e:
                    logger.warning(f"Error normalizing overall assessment score: {e}")
                    summary["overall_assessment"] = "Model Assessment: N/A"

            if "market_fit_analysis" in summary:
                if isinstance(summary["market_fit_analysis"], dict):
                    score = summary["market_fit_analysis"].get("score")
                    if score not in ["N/A", None]:
                        try:
                            normalized_score = self.normalize_scores([float(score)])[0]
                            summary["market_fit_analysis"]["score"] = f"{normalized_score:.2f}"
                        except ValueError as e:
                            logger.warning(f"Error normalizing market fit score: {e}")
                            summary["market_fit_analysis"]["score"] = "N/A"
                else:
                    summary["market_fit_analysis"] = {"score": "N/A", "justification": "Invalid market fit analysis data"}

            # Ensure consistent formatting for risk factors and recommendations
            if "risk_factors" in summary:
                for risk in summary["risk_factors"]:
                    risk["impact"] = risk.get("impact", "Medium")
                    risk["probability"] = risk.get("probability", "Medium")
                    risk["description"] = risk.get("description", "No description provided")
                    risk["recommendation"] = risk.get("recommendation", "No specific recommendation provided")

            if "recommendations" in summary:
                for rec in summary["recommendations"]:
                    rec["priority"] = rec.get("priority", "Medium")
                    rec["description"] = rec.get("description", "No description provided")

            # Ensure focus_areas and warnings are not empty
            if not summary["focus_areas"]:
                summary["focus_areas"] = ["No specific focus areas identified"]

            if not summary["warnings"]:
                summary["warnings"] = ["No warnings reported"]

            # 既存のコードに加えて、以下の処理を追加
            if not summary["key_insights"]:
                summary["key_insights"].append("No key insights could be extracted from the analysis")
            if not summary["risk_factors"]:
                summary["risk_factors"].append({
                    "risk_type": "Unknown",
                    "description": "No specific risks identified in the analysis",
                    "mitigation_strategy": "Further analysis recommended"
                })
            if not summary["recommendations"]:
                summary["recommendations"].append({
                    "strategy": "Further Analysis",
                    "description": "Insufficient data to provide specific recommendations",
                    "impact_score": 0,
                    "implementation_steps": "Gather more data and conduct a detailed analysis"
                })
            if summary["market_fit_analysis"]["score"] == "N/A" and not summary["market_fit_analysis"]["justification"]:
                summary["market_fit_analysis"]["justification"] = "Insufficient data to assess market fit"
        
            return summary


        except Exception as e:
            logger.error(f"Error in ensure_data_consistency: {str(e)}", exc_info=True)
            return summary  # Return original summary if an error occurs

    def extract_key_insights(self, key_metrics: List[Dict[str, Any]]) -> List[str]:
        insights = []
        for metric in key_metrics:
            metric_name = metric.get('metric_name', 'Unknown Metric')
            value = metric.get('value', 'N/A')
            description = metric.get('description', 'No description provided')
            
            insight = f"{metric_name}: {value} - {description}"
            
            # Add additional analysis based on the metric
            if 'inflation' in metric_name.lower():
                if value != 'N/A':
                    try:
                        inflation_rate = float(value.rstrip('%'))
                        if inflation_rate > 5:
                            insight += " This relatively high inflation rate may lead to token value dilution over time."
                        elif inflation_rate < 2:
                            insight += " This low inflation rate may limit token supply growth, potentially affecting network participation."
                    except ValueError:
                        pass
            elif 'staking' in metric_name.lower():
                if value != 'N/A':
                    insight += " Staking rewards can incentivize long-term holding and network participation."
                else:
                    insight += " The lack of specified staking rewards may impact token holder engagement."
            
            insights.append(insight)
        
        return insights if insights else ["No key insights could be extracted from the analysis"]


    # 1. サマリー生成ロジックの強化 - フォーカスエリア分析
    def analyze_focus_areas(self, focus_areas: List[str]) -> str:
        unique_areas = set(focus_areas)
        if not unique_areas:
            return "No specific focus areas were identified in the analysis. This could indicate a broad, general approach or a lack of clear focal points in the token economics model."

        analysis = "The analysis focused on the following areas: " + ", ".join(unique_areas) + ". "
        if len(unique_areas) > 1:
            analysis += "This diverse focus provides a comprehensive view of the token economics model, "
            analysis += f"covering aspects such as {', '.join(list(unique_areas)[:3])}. "
            analysis += "Such a multi-faceted approach allows for a holistic understanding of the token ecosystem."
        else:
            single_area = next(iter(unique_areas))
            analysis += f"This focused approach provides a deep dive into {single_area}, "
            analysis += "allowing for a detailed examination of this specific aspect of the token economics model. "
            analysis += f"While this concentration on {single_area} offers valuable insights, "
            analysis += "it's important to consider how it interacts with other aspects of the token ecosystem."

        analysis += " Future analyses may benefit from "
        analysis += "expanding the focus to include additional areas" if len(unique_areas) == 1 else "maintaining this comprehensive approach"
        analysis += " to ensure a well-rounded understanding of the token economics."

        # Add frequency analysis
        area_frequency = {area: focus_areas.count(area) for area in unique_areas}
        most_common = max(area_frequency, key=area_frequency.get)
        analysis += f" The most frequently mentioned focus area was '{most_common}', "
        analysis += f"which was highlighted by {area_frequency[most_common]} out of {len(focus_areas)} analyses."

        return analysis


    def process_key_metrics(self, key_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_metrics = []
        for metric in key_metrics:
            if isinstance(metric, dict):
                metric_name = metric.get('metric_name', metric.get('metric', 'Unknown metric'))
                value = metric.get('value', 'N/A')
                description = metric.get('description', 'No description provided')
                # 数値データを文字列に変換
                if isinstance(value, (int, float)):
                    value = f"{value:.2f}" if isinstance(value, float) else str(value)
                processed_metrics.append({
                    'metric_name': metric_name,
                    'value': value,
                    'description': description
                })
            elif isinstance(metric, str):
                processed_metrics.append({'metric_name': 'Unknown', 'value': metric, 'description': 'No description provided'})
        return processed_metrics


    def process_key_insights(self, key_metrics: List[Dict[str, Any]]) -> List[str]:
        processed_insights = []
        metric_details = defaultdict(list)

        for metric in key_metrics:
            if isinstance(metric, dict):
                metric_name = metric.get('metric_name', 'Unknown metric')
                value = metric.get('value', 'N/A')
                description = metric.get('description', '')
                metric_details[metric_name].append({
                    'value': value,
                    'description': description
                })
            elif isinstance(metric, str):
                processed_insights.append(metric)

        for metric_name, details in metric_details.items():
            values = set(detail['value'] for detail in details)
            descriptions = set(detail['description'] for detail in details if detail['description'])
            insight = f"{metric_name}: {', '.join(values)}"
            if descriptions:
                insight += f" - {' '.join(descriptions)}"

            # 追加の分析や洞察を生成
            if metric_name.lower() in ['annual inflation rate', 'inflation rate']:
                value = next(iter(values), 'N/A')
                if value != 'N/A':
                    try:
                        inflation_rate = float(value.rstrip('%'))
                        if inflation_rate > 5:
                            insight += f" High inflation rate may lead to token value dilution. Consider implementing deflationary mechanisms or adjusting the rate based on token demand and network growth."
                        elif inflation_rate < 2:
                            insight += f" Low inflation rate may limit token supply growth. Evaluate if this aligns with the project's long-term goals and consider potential impacts on token distribution and network participation."
                        else:
                            insight += f" The inflation rate appears balanced. Regularly monitor its effects on token value and adjust as needed to maintain economic stability."
                    except ValueError:
                        pass
            elif metric_name.lower() in ['staking reward', 'staking rewards']:
                value = next(iter(values), 'N/A')
                if value != 'N/A':
                    try:
                        staking_reward = float(value.rstrip('%'))
                        if staking_reward > 10:
                            insight += f" High staking rewards may attract more participants but could be unsustainable long-term. Consider implementing a dynamic reward system that adjusts based on network participation and token economics."
                        elif staking_reward < 3:
                            insight += f" Low staking rewards might not provide sufficient incentive for participation. Evaluate the impact on network security and consider additional benefits for stakers to increase participation."
                        else:
                            insight += f" The staking reward rate appears balanced. Monitor its effectiveness in attracting and retaining participants, and adjust as needed based on network growth and competition."
                    except ValueError:
                        pass
                else:
                    insight += f" Staking rewards are not specified. Defining clear staking rewards is crucial for incentivizing network participation and token holding. Consider implementing a transparent and competitive reward structure."

            processed_insights.append(insight)

        return self.remove_duplicates_and_similar(processed_insights)


    def process_risks_and_recommendations(self, risks: List[Dict[str, Any]], recommendations: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        unique_risks = self.remove_duplicates_and_similar(risks)
        unique_recommendations = self.remove_duplicates_and_similar(recommendations)

        final_risks = []
        final_recommendations = []

        for risk in unique_risks:
            risk_description = risk.get('description', '').lower()
            related_recommendation = next((rec for rec in unique_recommendations if self.is_related(risk_description, rec.get('strategy', '').lower())), None)
            
            if related_recommendation:
                combined_item = {
                    "risk_type": risk.get('risk_type', 'Unknown Risk'),
                    "description": risk.get('description', 'No description provided'),
                    "mitigation_strategy": related_recommendation.get('strategy', 'No strategy provided'),
                    "impact_score": related_recommendation.get('impact_score', 0),
                    "recommendation": related_recommendation.get('description', 'No recommendation provided')
                }
                final_risks.append(combined_item)
                unique_recommendations.remove(related_recommendation)
            else:
                risk['recommendation'] = 'No specific recommendation provided'
                final_risks.append(risk)

        final_recommendations = [{
            "strategy": rec.get('strategy', 'No strategy provided'),
            "description": rec.get('description', 'No description provided'),
            "impact_score": rec.get('impact_score', 0)
        } for rec in unique_recommendations]

        if not final_risks:
            final_risks.append({
                "risk_type": "Unknown",
                "description": "No specific risks identified in the analysis",
                "recommendation": "Further analysis recommended"
            })

        if not final_recommendations:
            final_recommendations.append({
                "strategy": "Further Analysis",
                "description": "Insufficient data to provide specific recommendations"
            })

        return final_risks, final_recommendations



    def remove_circular_references(self, data):
        if isinstance(data, dict):
            return {k: self.remove_circular_references(v) for k, v in data.items() if k != 'previous_responses'}
        elif isinstance(data, list):
            return [self.remove_circular_references(v) for v in data]
        else:
            return data

    def preprocess_for_storage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # 循環参照を解消
        data = self.remove_circular_references(data)
        
        # 大きすぎるフィールドの処理
        max_field_size = 16793600  # MongoDB のドキュメントサイズ制限: 16MB
        for key, value in data.items():
            if isinstance(value, str) and len(value.encode('utf-8')) > max_field_size:
                data[key] = value[:max_field_size // 2]  # 簡易的な切り詰め
                logger.warning(f"Field '{key}' was truncated due to size limitations")
        
        return data

    def is_related(self, text1: str, text2: str, threshold: float = 0.3) -> bool:
        return SequenceMatcher(None, text1, text2).ratio() > threshold
                
    
    def remove_duplicates_and_similar(self, items: List[Any]) -> List[Any]:
        unique_items = []
        for item in items:
            if not self.is_similar_to_existing(item, unique_items):
                unique_items.append(item)
        return unique_items


    def is_similar_to_existing(self, item: Any, existing_items: List[Any], threshold: float = 0.8) -> bool:
        item_str = json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item)
        for existing in existing_items:
            existing_str = json.dumps(existing, sort_keys=True) if isinstance(existing, dict) else str(existing)
            similarity = SequenceMatcher(None, item_str, existing_str).ratio()
            if similarity >= threshold:
                return True
        return False

    def synthesize_justifications(self, justifications: List[str]) -> str:
        # 最も長い justification を選択し、他の justification からキーワードを抽出して追加
        main_justification = max(justifications, key=len)
        keywords = set()
        for justification in justifications:
            keywords.update(word.lower() for word in justification.split() if len(word) > 3)
        additional_info = ", ".join(sorted(keywords)[:5])  # 上位5つのキーワードを追加
        return f"{main_justification} Additional key aspects: {additional_info}"



    def process_market_fit_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        market_fit_scores = []
        justifications = []

        for result in results:
            content = result.get("content", {})
            if isinstance(content, dict):
                market_fit = content.get("market_fit_analysis", {})
                if isinstance(market_fit, dict):
                    score = market_fit.get("score")
                    justification = market_fit.get("justification")
                    if score is not None:
                        market_fit_scores.append(float(score))
                    if justification:
                        justifications.append(justification)

        if market_fit_scores:
            average_score = sum(market_fit_scores) / len(market_fit_scores)
        else:
            average_score = None

        combined_justification = self.synthesize_justifications(justifications) if justifications else "Insufficient data to provide a justification."

        return {
            "score": average_score,
            "justification": combined_justification
        }

    def normalize_scores(self, scores: List[float], min_value: float = 0, max_value: float = 100) -> List[float]:
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if min_score == max_score:
            return [max_value] * len(scores)
        
        return [
            ((score - min_score) / (max_score - min_score)) * (max_value - min_value) + min_value
            for score in scores
        ]

    def remove_circular_references(self, data):
        if isinstance(data, dict):
            return {k: self.remove_circular_references(v) for k, v in data.items() if k != 'previous_responses'}
        elif isinstance(data, list):
            return [self.remove_circular_references(v) for v in data]
        else:
            return data


    # 1. サマリー生成ロジックの強化 - 重要な洞察の抽出
    def extract_key_insights(self, key_metrics: List[Dict[str, Any]]) -> List[str]:
        """
        重要な洞察を抽出する関数
        """
        insights = []
        for metric in key_metrics:
            metric_name = metric.get('metric_name', 'Unknown Metric')
            value = metric.get('value', 'N/A')
            description = metric.get('description', 'No description provided')
            
            insight = f"{metric_name}: {value} - {description}"
            
            # メトリクスに基づいた追加の分析
            if 'inflation' in metric_name.lower():
                if value != 'N/A':
                    try:
                        inflation_rate = float(value.rstrip('%'))
                        if inflation_rate > 5:
                            insight += " This relatively high inflation rate may lead to token value dilution over time."
                        elif inflation_rate < 2:
                            insight += " This low inflation rate may limit token supply growth, potentially affecting network participation."
                    except ValueError:
                        pass
            elif 'staking' in metric_name.lower():
                if value != 'N/A':
                    insight += " Staking rewards can incentivize long-term holding and network participation."
                else:
                    insight += " The lack of specified staking rewards may impact token holder engagement."
            
            insights.append(insight)
        
        return insights if insights else ["No key insights could be extracted from the analysis"]

    def integrate_market_fit_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        market_fit_scores = []
        justifications = []
        for result in results:
            content = result.get("content", {})
            if isinstance(content, dict):
                market_fit = content.get("market_fit_analysis", {})
                if isinstance(market_fit, dict):
                    score = market_fit.get("score")
                    justification = market_fit.get("justification")
                    if score is not None:
                        market_fit_scores.append(float(score))
                    if justification:
                        justifications.append(justification)

        if market_fit_scores:
            average_score = sum(market_fit_scores) / len(market_fit_scores)
        else:
            average_score = 0

        combined_justification = self.synthesize_justifications(justifications) if justifications else "Insufficient data to provide a justification."

        return {
            "score": average_score,
            "justification": combined_justification
        }

    def extract_focus_areas(self, focus_areas: List[str]) -> List[str]:
        """
        フォーカスエリアを抽出する関数
        """
        return list(set(focus_areas)) if focus_areas else ["No specific focus areas identified"]

        # 重複を削除し、空の文字列を除外
        unique_areas = list(set(area.strip() for area in focus_areas if area.strip()))
    
        if not unique_areas:
            return ["No valid focus areas identified"]
    
        # フォーカスエリアを重要度順にソート（仮定：出現頻度が高いものほど重要）
        sorted_areas = sorted(unique_areas, key=lambda x: focus_areas.count(x), reverse=True)
    
        return sorted_areas[:5]  # 上位5つのフォーカスエリアを返す


    # 7. 重み付け評価システムの実装
    def apply_weighted_evaluation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        各エージェントの結果に重み付けを適用する関数
        """
        weighted_results = {}
        total_weight = sum(self.agent_weights.values())

        for result in results:
            agent_name = result.get("sender", "Unknown Agent")
            weight = self.agent_weights.get(agent_name, 1)  # デフォルトの重みは1
            normalized_weight = weight / total_weight

            for key, value in result.get("content", {}).items():
                if isinstance(value, (int, float)):
                    weighted_results[key] = weighted_results.get(key, 0) + (value * normalized_weight)
                elif isinstance(value, list):
                    weighted_results[key] = weighted_results.get(key, []) + value
                elif isinstance(value, dict):
                    if key not in weighted_results:
                        weighted_results[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            weighted_results[key][sub_key] = weighted_results[key].get(sub_key, 0) + (sub_value * normalized_weight)
                        else:
                            weighted_results[key][sub_key] = sub_value

        return weighted_results

    # 8. キャッシュメカニズムの実装
    @lru_cache(maxsize=100)
    def cached_calculation(self, input_data: str) -> Any:
        """
        頻繁に使用される計算結果をキャッシュする関数
        """
        # このアプリケーションに合わせた実装
        try:
            data = json.loads(input_data)
            # 結果の数をカウント
            result_count = len(data)
            # 全ての model_assessment スコアの平均を計算
            model_assessments = [
                float(result['content']['model_assessment'])
                for result in data
                if 'content' in result and 'model_assessment' in result['content']
            ]
            average_assessment = sum(model_assessments) / len(model_assessments) if model_assessments else 0
            
            return {
                "result_count": result_count,
                "average_assessment": average_assessment
            }
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON input: {input_data}")
            return None
        except Exception as e:
            logger.error(f"Error in cached_calculation: {str(e)}")
            return None


    # 新しい関数: 重み付けされた結果の分析
    def analyze_weighted_results(self, weighted_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        重み付けされた結果を分析する関数
        """
        analysis = {}
        for key, value in weighted_results.items():
            if isinstance(value, (int, float)):
                analysis[key] = f"Weighted average: {value:.2f}"
            elif isinstance(value, dict):
                analysis[key] = {sub_key: f"Weighted average: {sub_value:.2f}" for sub_key, sub_value in value.items() if isinstance(sub_value, (int, float))}
        return analysis

    def calculate_overall_assessment(self, summary: Dict[str, Any]) -> str:
        """
        全体評価を計算する関数（改善版）
        """
        model_assessments = []
        for result in summary.get("agent_specific_analysis", {}).values():
            assessment = result.get("model_assessment")
            if isinstance(assessment, (int, float, str)):
                try:
                    model_assessments.append(float(assessment))
                except ValueError:
                    logger.warning(f"Invalid model assessment value: {assessment}")

        if model_assessments:
            average_score = sum(model_assessments) / len(model_assessments)
            return f"Model Assessment: {average_score:.2f}/100"
        else:
            return "Unable to determine overall assessment due to insufficient data"


async def main():
    workflow_manager = WorkflowManager()
    token_data = {
        "token_supply": 1000000,
        "initial_price": 0.1,
        "inflation_rate": 0.05,
        "staking_reward": 0.08,
        "burn_mechanism": "2% of transaction fees",
        "vesting_schedule": "10% unlock per quarter"
    }
    result = await workflow_manager.run_token_economics_analysis(token_data)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
