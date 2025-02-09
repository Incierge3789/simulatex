import logging
import json
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline, AutoTokenizer
import torch
import time
import traceback

# ロギングの設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ファイルハンドラを追加
file_handler = logging.FileHandler('response_processor.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# JSON形式のログ記録を追加
def log_json(message, level=logging.INFO, extra=None):
    log_data = {
        "timestamp": time.time(),
        "level": logging.getLevelName(level),
        "message": message
    }
    if extra:
        log_data.update(extra)
    logger.log(level, json.dumps(log_data))

# GPUが利用可能かどうかを確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_json(f"Using device: {device}", level=logging.INFO)

async def remove_duplicates(texts, threshold=0.8):
    try:
        log_json("Starting remove_duplicates function", level=logging.DEBUG, extra={"num_texts": len(texts)})
        start_time = time.time()
        vectorizer = TfidfVectorizer().fit_transform(texts)
        cosine_similarities = cosine_similarity(vectorizer)
        
        unique_texts = []
        for i, text in enumerate(texts):
            if all(cosine_similarities[i][j] < threshold for j in range(i)):
                unique_texts.append(text)
        
        execution_time = time.time() - start_time
        log_json("Finished remove_duplicates function", level=logging.INFO, 
                 extra={"removed_duplicates": len(texts) - len(unique_texts), "execution_time": execution_time})
        return unique_texts
    except Exception as e:
        log_json("Error in remove_duplicates", level=logging.ERROR, 
                 extra={"error": str(e), "traceback": traceback.format_exc()})
        return texts  # エラーが発生した場合は元のテキストリストを返す

def calculate_max_length(text):
    # 入力テキストの長さに基づいて適切なmax_lengthを計算
    return min(max(30, len(text) // 3), 500)  # 最小30、最大500、デフォルトはテキスト長の1/3

async def summarize_text(text, max_length=None, min_length=None):
    try:
        if not text:
            log_json("Empty input text", level=logging.WARNING)
            return "Empty input text"

        log_json("Starting summarize_text function", level=logging.DEBUG, extra={"text_length": len(text)})
        start_time = time.time()

        if max_length is None:
            max_length = calculate_max_length(text)
        if min_length is None:
            min_length = max(30, max_length // 3)  # max_lengthの1/3か30のいずれか大きい方

        # 警告メッセージに対応するためのチェック
        if max_length <= min_length:
            log_json("max_length is less than or equal to min_length", level=logging.WARNING)
            max_length = min_length + 1

        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        summarizer = pipeline("summarization", model=model_name, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        
        # テキストを適切な長さに分割
        chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
        summaries = []

        for chunk in chunks:
            summary = await asyncio.to_thread(summarizer, chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(summary[0]['summary_text'])

        final_summary = " ".join(summaries)
        
        # 要約品質のチェック
        if len(final_summary) < min_length:
            log_json("Summary is too short. Regenerating...", level=logging.WARNING)
            final_summary = await asyncio.to_thread(summarizer, text, max_length=max_length, min_length=min_length, do_sample=True)
            final_summary = final_summary[0]['summary_text']
        
        execution_time = time.time() - start_time
        log_json("Finished summarize_text function", level=logging.INFO, 
                 extra={"summary_length": len(final_summary), "execution_time": execution_time})
        return final_summary
    except Exception as e:
        log_json("Error in summarize_text", level=logging.ERROR, 
                 extra={"error": str(e), "traceback": traceback.format_exc()})
        return f"Error summarizing text: {str(e)}"

async def process_responses(analyst_response, gemini_response, claude_response, gpt4_response):
    try:
        log_json("Starting process_responses function", level=logging.INFO)
        start_time = time.time()

        # 入力データの妥当性チェック
        responses = [analyst_response, gemini_response, claude_response, gpt4_response]
        valid_responses = [response for response in responses if isinstance(response, str)]

        if len(valid_responses) != len(responses):
            log_json("Some inputs are not strings", level=logging.WARNING)

        processed_responses = []

        # 重複の削除
        unique_responses = await remove_duplicates(valid_responses)
        log_json("Removed duplicates", level=logging.INFO, extra={"unique_responses": len(unique_responses)})

        async def process_response(i, response):
            try:
                log_json(f"Processing response {i+1}", level=logging.DEBUG)
                # 要約の生成
                summary = await summarize_text(response)
                return {
                    "original": response,
                    "summary": summary,
                    "word_count": len(response.split())
                }
            except Exception as e:
                log_json(f"Error processing response {i+1}", level=logging.ERROR, 
                         extra={"error": str(e), "traceback": traceback.format_exc()})
                return {
                    "original": response,
                    "error": f"Error processing response {i+1}: {str(e)}"
                }

        # 非同期でレスポンスを処理
        tasks = [process_response(i, response) for i, response in enumerate(unique_responses)]
        processed_responses = await asyncio.gather(*tasks)

        execution_time = time.time() - start_time
        log_json("Finished process_responses function", level=logging.INFO, 
                 extra={"num_processed": len(processed_responses), "execution_time": execution_time})
        return processed_responses

    except Exception as e:
        log_json("Error in process_responses", level=logging.ERROR, 
                 extra={"error": str(e), "traceback": traceback.format_exc()})
        return [{"error": f"An unexpected error occurred during processing: {str(e)}"}]

async def main():
    log_json("Starting main function", level=logging.INFO)
    # テスト用のダミーレスポンス
    test_responses = [
        "This is a test response from Analyst AI.",
        "This is a test response from Gemini.",
        "This is a test response from Claude.",
        "This is a test response from GPT-4."
    ]
    
    try:
        result = await asyncio.wait_for(process_responses(*test_responses), timeout=60)  # 60秒のタイムアウトを設定
        log_json("Finished processing responses", level=logging.INFO, extra={"result": result})
        print(result)
    except asyncio.TimeoutError:
        log_json("Processing timed out", level=logging.ERROR)
        print("Processing timed out")
    except Exception as e:
        log_json("Error in main function", level=logging.ERROR, 
                 extra={"error": str(e), "traceback": traceback.format_exc()})
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
