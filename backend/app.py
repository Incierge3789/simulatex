from concurrent.futures import ThreadPoolExecutor  # 追加
from flask import Flask, request, jsonify
from flask_cors import CORS
from communication_protocol import CommunicationProtocol
from workflow_manager import WorkflowManager
from data_manager import DataManager
import logging
import os
import asyncio
from asgiref.sync import async_to_sync
from flask import g
import time
from agents.gpt4_agent import GPT4Agent
from agents.claude_agent import ClaudeAgent
from agents.gemini_agent import GeminiAgent
from agents.analyst_ai import AnalystAI

app = Flask(__name__)
CORS(app)

# ロギングの設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AppState:
    def __init__(self):
        # # 基本設定
        self._loop = None
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.communication_protocol = CommunicationProtocol()
        self.workflow_manager = WorkflowManager()
        
        # エージェントの初期化
        self.gpt4_agent = GPT4Agent()
        self.claude_agent = ClaudeAgent()
        self.gemini_agent = GeminiAgent()
        self.analyst_ai = AnalystAI()
        self.initialize_agents()

    def initialize_agents(self):
        """エージェントの初期化と登録"""
        self.agents = {
            'analyst_ai': self.analyst_ai.query_analyst_ai,
            'claude': self.claude_agent.query_claude,
            'gemini': self.gemini_agent.query_gemini,
            'gpt4': self.gpt4_agent.query_gpt4
        }

        # エージェントの登録
        for agent_name, query_func in self.agents.items():
            try:
                self.communication_protocol.register_agent(agent_name, query_func)
            except Exception as e:
                logger.error(f"Error registering agent {agent_name}: {str(e)}")

    async def ensure_initialized(self):
        """コンポーネントの初期化を確認"""
        if self.data_manager is None:
            self.data_manager = DataManager()
            # # DataManagerの初期化を確認
            await self.data_manager.ensure_initialized()

    async def get_loop(self):
        """イベントループの取得"""
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

app_state = AppState()


async def process_agents_sequentially(user_message):
    """順次実行: 各エージェントを順番に処理"""
    responses = {}
    agents_order = ['analyst_ai', 'claude', 'gemini', 'gpt4']
    
    try:
        # 現在のイベントループを取得
        loop = await app_state.get_loop()
        
        for agent in agents_order:
            try:
                logger.info(f"Processing agent: {agent}")
                
                message = {
                    "sender": "System",
                    "receiver": agent,
                    "message_type": "analysis_request",
                    "content": {
                        "user_message": user_message,
                        "previous_responses": responses
                    }
                }
                
                # 同じループで非同期処理を実行
                response = await app_state.communication_protocol.route_message(message)
                
                if response['message_type'] == 'error':
                    logger.warning(f"Error from {agent}: {response['content']['error_message']}")
                else:
                    responses[agent.lower().replace(' ', '_')] = response['content']
                    
            except Exception as e:
                logger.error(f"Error processing {agent}: {str(e)}", exc_info=True)
                
        return responses
    except Exception as e:
        logger.error(f"Error in process_agents_sequentially: {str(e)}", exc_info=True)
        raise

@app.route('/api/team', methods=['POST'])
def virtual_team():
    try:
        user_message = request.json.get('message')
        execution_mode = request.json.get('execution_mode', 'sequential')

        async def process_request():
            try:
                # # 同じループを使用
                loop = await app_state.get_loop()
                
                if execution_mode == 'parallel':
                    responses = await process_agents_in_parallel(user_message)
                else:
                    responses = await process_agents_sequentially(user_message)

                if not responses:
                    return {"error": "All agents failed to process the request"}, 500

                # # WorkflowManagerの処理を同じループで実行
                workflow_result = await app_state.workflow_manager.run_token_economics_analysis({
                    "user_message": user_message,
                    "agent_responses": responses
                })
                return workflow_result

            except Exception as e:
                logger.error(f"Error in async processing: {str(e)}", exc_info=True)
                raise

        # # イベントループの管理を改善
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(process_request())
            return jsonify(result)
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Error in virtual_team: {str(e)}", exc_info=True)
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e)
        }), 500

async def process_agents_in_parallel(user_message):
    """
    並列実行: 全エージェントを同時に処理
    """
    async def process_agent(agent):
        try:
            logger.info(f"Processing agent: {agent}")
            start_time = asyncio.get_event_loop().time()

            # メッセージ作成の修正
            message = {
                "sender": "System",
                "receiver": agent,
                "message_type": "analysis_request",
                "content": {
                    "user_message": user_message,
                    "previous_responses": {}
                }
            }
            response = await communication_protocol.route_message(message)

            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time
            logger.info(f"Agent {agent} processing time: {processing_time:.2f} seconds")

            if response['message_type'] == 'error':
                logger.warning(f"Error from {agent}: {response['content']['error_message']}")
                return {agent: response['content']['error_message']}
            else:
                return {agent.lower().replace(' ', '_'): response['content']}
        except Exception as e:
            logger.error(f"Error processing {agent}: {str(e)}", exc_info=True)
            return {agent: str(e)}

    results = await asyncio.gather(*[process_agent(agent) for agent in agents.keys()])
    
    responses = {}
    errors = []
    for result in results:
        if result:
            if isinstance(result, dict):
                responses.update(result)
            else:
                errors.append(result)
    
    if errors:
        logger.warning(f"Errors during parallel processing: {errors}")
    
    return responses

@app.route('/api/search', methods=['GET'])
async def search():
    try:
        query = request.args.get('query', '')
        results = await data_manager.find_similar_analyses(query)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in search: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred during search.", "details": str(e)}), 500

@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    diff = time.time() - g.start_time
    logger.info(f"Request processed in {diff:.2f} seconds")
    return response


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 42069))
    app.run(debug=True, host='0.0.0.0', port=port)
