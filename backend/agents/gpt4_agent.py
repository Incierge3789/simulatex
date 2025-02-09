import asyncio
import datetime
import json
import logging
import os
import random
import time
import traceback

import tiktoken
from communication_protocol import CommunicationProtocol
from dotenv import load_dotenv
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("gpt4_agent.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

load_dotenv()


class GPT4Agent:
    def __init__(self):
        self.agent_name = "gpt4"
        self.communication_protocol = CommunicationProtocol()
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
            "long-term sustainability",
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

    def num_tokens_from_string(self, string: str, model_name: str) -> int:
        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def truncate_or_summarize_input(self, input_data: dict, max_tokens: int) -> dict:
        input_str = json.dumps(input_data)
        if self.num_tokens_from_string(input_str, "gpt-4") <= max_tokens:
            return input_data
        truncated_data = {}
        total_tokens = 0
        for key, value in input_data.items():
            value_str = str(value)
            value_tokens = self.num_tokens_from_string(value_str, "gpt-4")
            if total_tokens + value_tokens <= max_tokens:
                truncated_data[key] = value
                total_tokens += value_tokens
            else:
                break
        return truncated_data

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

    async def query_gpt4(self, message: dict):
        start_time = time.time()
        receiver = message.get("sender", "Unknown")
        try:
            self.log_json(
                "INFO",
                "Starting GPT-4 query",
                {"message_type": message.get("message_type")},
            )
            input_data = message.get("content", {})
            input_data = self.validate_and_enrich_input_data(input_data)
            input_data = self.standardize_data(input_data)
            max_input_tokens = 3000
            truncated_input = self.truncate_or_summarize_input(
                input_data, max_input_tokens
            )
            unique_focus = self.generate_unique_prompt()
            prompt = self.generate_analysis_prompt(truncated_input, unique_focus)
            response = await self.get_gpt4_response(prompt)
            gpt4_response = json.loads(response.choices[0].message.content)
            standardized_response = self.process_gpt4_response(
                gpt4_response, unique_focus
            )
            execution_time = time.time() - start_time
            standardized_response["execution_time"] = execution_time
            return await self.communication_protocol.create_message(
                sender=self.agent_name,
                receiver=receiver,
                message_type="token_economics_analysis",
                content=standardized_response,
            )
        except Exception as e:
            return await self.handle_error(e, receiver, start_time, input_data)

    def generate_analysis_prompt(self, input_data, unique_focus):
        prompt = f"""Analyze the following token economics data and provide strategic insights. {unique_focus} Your response should be in a structured JSON format for efficient processing by other AI agents.

Input data:
{json.dumps(input_data, indent=2)}

Respond with a JSON object containing the following keys:
1. model_assessment: A numerical score (0-100) representing the overall quality of the token economics model.
2. key_metrics: An array of important metrics extracted from the input data, each with a 'metric_name', 'value', and 'description'.
3. risk_factors: An array of potential risks identified in the model, each with a 'risk_type', 'description', and 'mitigation_strategy'.
4. optimization_strategies: An array of strategies to improve the model, each with a 'strategy', 'impact_score' (0-100), and 'implementation_steps'.
5. innovation_proposals: An array of innovative features or approaches, each with a 'proposal', 'feasibility_score' (0-100), and 'potential_impact'.
6. market_fit_analysis: An object with 'score' (0-100) representing how well the model fits current market trends, 'justification' providing a brief explanation, and 'improvement_suggestions'.

Ensure your response is concise, focused on quantifiable data and actionable insights, and particularly addresses the specified focus area. Provide detailed descriptions for all fields, avoiding generic responses."""
        return prompt

    async def get_gpt4_response(self, prompt):
        total_tokens = self.num_tokens_from_string(prompt, "gpt-4")
        max_response_tokens = 8192 - total_tokens - 100
        try:
            return await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI specializing in token economics analysis. Provide responses in the specified JSON format, focusing on quantitative assessments and actionable insights.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_response_tokens,
                temperature=0.7,
            )
        except Exception as api_error:
            self.log_json("ERROR", "Error calling GPT-4 API", {"error": str(api_error)})
            raise

    def process_gpt4_response(self, gpt4_response, unique_focus):
        """GPT-4レスポンスの処理と標準化"""
        try:
            # デフォルト値の設定
            gpt4_response = self.set_default_values(gpt4_response)

            # レスポンスの標準化
            standardized_response = {
                "focus_area": unique_focus,  # トップレベルに設定
                "content": {
                    **self.standardize_data(gpt4_response),
                    "focus_area": unique_focus,  # content内にも設定
                    "analysis_type": "token_economics_analysis",
                },
                "metadata": {
                    "agent_name": self.agent_name,
                    "focus_area": unique_focus,  # メタデータ内にも設定
                    "timestamp": datetime.datetime.now().isoformat(),
                    "standardization_version": "1.0",
                    "processing_status": "completed",
                },
            }

            # 検証ログ
            self.log_json(
                "DEBUG",
                "Processed GPT-4 response",
                {
                    "has_focus_area": "focus_area" in standardized_response,
                    "focus_area": unique_focus,
                    "response_structure": list(standardized_response.keys()),
                },
            )

            return standardized_response

        except Exception as e:
            self.log_json(
                "ERROR",
                "Error processing GPT-4 response",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            return {
                "focus_area": unique_focus,
                "error": str(e),
                "metadata": {
                    "agent_name": self.agent_name,
                    "focus_area": unique_focus,
                    "error": True,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
            }

    async def handle_error(self, e, receiver, start_time, input_data):
        """エラー処理の改善"""
        error_message = f"Error in GPT-4 processing: {str(e)}"
        self.log_json("ERROR", error_message, {"traceback": traceback.format_exc()})

        execution_time = time.time() - start_time
        partial_results = self.extract_partial_results(input_data)

        error_response = {
            "focus_area": "error_handling",  # エラー時のfocus_area
            "error_code": "GPT4_PROCESSING_ERROR",
            "error_message": error_message,
            "execution_time": execution_time,
            "partial_results": partial_results,
        }

        return await self.communication_protocol.create_message(
            sender=self.agent_name,
            receiver=receiver,
            message_type="error",
            content=error_response,
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
                        for default_key, default_value in defaults.items():
                            if default_key not in item or item[default_key] is None:
                                item[default_key] = default_value
        return data


# クラスの外でmainを定義
async def main():
    """メインの実行関数"""
    try:
        agent = GPT4Agent()
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
            receiver="gpt4",
            message_type="token_economics_data",
            content=test_input,
        )

        result = await agent.query_gpt4(test_message)
        print(json.dumps(result, indent=2))

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
