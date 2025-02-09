import asyncio
import datetime
import json
import logging
import os
import random
import re
import time
import traceback

from anthropic import (
    APIError,
    APIStatusError,
    AsyncAnthropic,
    InternalServerError,
)
from communication_protocol import CommunicationProtocol
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("claude_agent.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


class ClaudeAgent:
    def __init__(self):
        self.agent_name = "claude"
        self.communication_protocol = CommunicationProtocol()
        self.base_timeout = 180
        self.max_retries = 3
        self.base_delay = 2

    def log_json(self, level, message, extra=None):
        log_data = {
            "timestamp": datetime.datetime.now().isoformat(),
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
            "regulatory compliance in different jurisdictions",
            "ethical implications of token distribution",
            "long-term sustainability and environmental impact",
            "social impact and inclusivity",
            "governance structure and decentralization",
            "interoperability with other blockchain ecosystems",
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

    def standardize_data(self, data):
        if isinstance(data, dict):
            return {k: self.standardize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.standardize_data(item) for item in data]
        elif isinstance(data, (int, float)):
            return float(data)
        elif isinstance(data, str):
            return data.strip()
        else:
            return str(data)

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
            "ethical_considerations",
        ]:
            if field in data and isinstance(data[field], list):
                for item in data[field]:
                    if isinstance(item, dict):
                        if "description" not in item or item["description"] is None:
                            item["description"] = "No description provided"
        return data

    async def query_claude(self, message: dict):
        start_time = time.time()
        receiver = message.get("sender", "Unknown")

        try:
            input_data = message.get("content", {})
            input_data = self.validate_and_enrich_input_data(input_data)
            input_data = self.standardize_data(input_data)
            unique_focus = self.generate_unique_prompt()
            prompt = self.generate_analysis_prompt(input_data, unique_focus)

            for attempt in range(self.max_retries):
                try:
                    timeout = self.base_timeout * (2**attempt)
                    delay = self.base_delay * (2**attempt) + random.uniform(0, 1)

                    self.log_json(
                        "DEBUG",
                        f"Attempt {attempt + 1}",
                        {"timeout": timeout, "delay": delay, "attempt": attempt + 1},
                    )

                    response = await asyncio.wait_for(
                        client.messages.create(
                            model="claude-3-opus-20240229",
                            max_tokens=4000,
                            temperature=0.7,
                            messages=[{"role": "user", "content": prompt}],
                        ),
                        timeout=timeout,
                    )

                    if (
                        not response
                        or not response.content
                        or not response.content[0].text
                    ):
                        raise ValueError("Empty response from Claude API")

                    response_text = response.content[0].text.strip()
                    claude_response = await self.parse_response(response_text)

                    standardized_response = self.process_claude_response(
                        claude_response, unique_focus
                    )
                    execution_time = time.time() - start_time
                    standardized_response["execution_time"] = execution_time

                    return await self.communication_protocol.create_message(
                        sender=self.agent_name,
                        receiver=receiver,
                        message_type="token_economics_risk_analysis",
                        content=standardized_response,
                    )

                except (APIStatusError, InternalServerError) as e:
                    if "overloaded_error" in str(e):
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(delay)
                            continue
                    raise

                except asyncio.TimeoutError:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(delay)
                        continue
                    raise TimeoutError(
                        f"Claude query timed out after {self.max_retries} attempts"
                    )

                except Exception as e:
                    self.log_json(
                        "ERROR", f"Error on attempt {attempt + 1}", {"error": str(e)}
                    )
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(delay)
                    continue

        except Exception as e:
            return await self.handle_error(e, receiver, start_time, input_data)

    async def parse_response(self, response_text: str) -> dict:
        """レスポンスのパース処理を分離"""
        try:
            # JSONブロックを探す
            json_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)

            if json_blocks:
                for json_str in json_blocks:
                    try:
                        return json.loads(json_str.strip())
                    except json.JSONDecodeError:
                        continue

            # 直接JSONとして解析を試みる
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                pass

            # 最後の手段としてJSONブロックを探す
            json_start = response_text.find("{")
            if json_start != -1:
                json_end = response_text.rfind("}") + 1
                if json_end > json_start:
                    try:
                        return json.loads(response_text[json_start:json_end])
                    except json.JSONDecodeError:
                        pass

            raise ValueError("No JSON object found in response")

        except Exception as e:
            self.log_json(
                "ERROR",
                "JSON parse error",
                {"error": str(e), "response_sample": response_text[:200]},
            )
            raise

    def generate_analysis_prompt(self, input_data, unique_focus):
        prompt = f"""Analyze the following token economics data, focusing on risk analysis, ethical considerations, and regulatory compliance. {unique_focus} Your response should be in a structured JSON format for efficient processing by other AI agents.

Input data:
{json.dumps(input_data, indent=2)}

Respond with a JSON object containing the following keys:
1. model_assessment: A numerical score (0-100) representing the overall quality of the token economics model.
2. key_metrics: An array of important metrics extracted from the input data, each with a 'metric_name', 'value', and 'description'.
3. risk_factors: An array of identified risks, each with a 'risk_type', 'description', and 'mitigation_strategy'.
4. optimization_strategies: An array of strategies to improve the model, each with a 'strategy', 'impact_score' (0-100), and 'implementation_steps'.
5. innovation_proposals: An array of innovative features or approaches, each with a 'proposal', 'feasibility_score' (0-100), and 'potential_impact'.
6. market_fit_analysis: An object with 'score' (0-100) representing how well the model fits current market trends, 'justification' providing a brief explanation, and 'improvement_suggestions'.
7. ethical_considerations: An array of ethical issues, each with an 'issue', 'description', and 'recommendation'.
8. regulatory_compliance: An object with 'compliant_aspects' and 'non_compliant_aspects', each being an array of relevant points with descriptions.

Ensure your response is comprehensive, focusing on critical analysis, ethical implications, and actionable recommendations for improving the token economics model's robustness and compliance. Pay particular attention to the specified focus area. Provide detailed descriptions for all fields, avoiding generic responses."""

        return prompt

    def process_claude_response(self, claude_response, unique_focus):
        claude_response = self.set_default_values(claude_response)
        standardized_response = self.standardize_data(claude_response)
        standardized_response["focus_area"] = unique_focus
        return standardized_response

    async def handle_error(self, e, receiver, start_time, input_data):
        error_message = f"Error in Claude processing: {str(e)}"
        self.log_json("ERROR", error_message, {"traceback": traceback.format_exc()})

        execution_time = time.time() - start_time
        partial_results = self.extract_partial_results(input_data)

        return await self.communication_protocol.create_message(  # create_messageを使用
            sender=self.agent_name,
            receiver=receiver,
            message_type="error",
            content={
                "error_code": "CLAUDE_PROCESSING_ERROR",
                "error_message": error_message,
                "execution_time": execution_time,
                "partial_results": partial_results,
                "focus_area": "error_handling",
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

    async def main():
        try:
            agent = ClaudeAgent()
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
            test_message = await comm_protocol.create_message_async(
                sender="System",
                receiver="claude",
                message_type="token_economics_data",
                content=test_input,
            )

            # Claudeへのクエリ実行
            result = await agent.query_claude(test_message)

            # 結果の出力
            print(json.dumps(result, indent=2))

            return result  # 結果を返す

        except Exception as e:
            logger.error(f"Error in main: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    if __name__ == "__main__":
        asyncio.run(main())
