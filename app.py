import asyncio
import base64
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import flask
import numpy as np
import portalocker
import boto3
import json

import sounddevice
import speech_recognition as sr
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptResultStream, TranscriptEvent
from flask import Flask, render_template, Response, send_from_directory, request, jsonify, send_file
from flask_socketio import SocketIO
import threading
import time
from selenium.webdriver.chrome.service import Service
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
from api_request_schema import api_request_list
import os
import markdown
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
import sys
import uuid
from pathlib import Path
import fitz # pip install PyMuPDF
import re
from flask_cors import CORS # pip install flask-cors
import io
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# 初始化
app = Flask(__name__)
CORS(app)  # ⬅️ 允许 JS 跨域请求 MP3 流
app.config.update(
    UPLOAD_FOLDER = 'uploads',          # 上传目录
    ALLOWED_EXTENSIONS = {'png', 'jpeg', 'gif', 'webp', 'pdf'},  # 允许的文件类型
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 限制5MB大小
)
socketio = SocketIO(app, cors_allowed_origins="*")
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")# 初始化 AWS Bedrock 客户端

# 可调设置参数
class Config:
    # 超参数
    refresh_rag_documents = True
    voiceIndex = 0 # 语音代码：0:中文，1:英文，2:日文，3:法语
    audio = False  # 默认是否语音输出
    temperature = 0.1  # 默认温度
    embedding_model_name_index = 0  # 嵌入模型名称
    use_RAG = False
    model_index = 0
    model_id_list = ['anthropic.claude-3-5-sonnet-20240620-v1:0','anthropic.claude-3-sonnet-20240229-v1:0','meta.llama3-70b-instruct-v1']
    model_id = os.getenv('MODEL_ID', model_id_list[model_index])
    api_request = api_request_list[model_id]

    # 不修改参数
    embedding_model_names = ['BAAI/bge-reranker-base', 'BAAI/bge-large-en-v1.5', 'BAAI/bge-large-zh-v1.5', 'BAAI/bge-m3', 'Alibaba-NLP/gte-multilingual-base', 'ibm-granite/granite-embedding-278m-multilingual']
    SETTINGS_FILE = 'static/settings/saved_settings.json'
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    voiceList = ["zh-CN", "en-US", "ja-JP", "fr-FR"]
    voiceLanguageList = ['cmn-CN', 'en-US', 'ja-JP', 'fr-FR']
    voiceNameList = ['Zhiyu', 'Ivy', 'Takumi', 'Remi']
    config = {
        'log_level': 'none',
        'region': aws_region,
        'polly': {
            'Engine': 'neural',
            'LanguageCode': voiceLanguageList[voiceIndex],
            'VoiceId': voiceNameList[voiceIndex],
            'OutputFormat': 'mp3',
        },
        'bedrock': {
            'api_request': api_request
        }
    }
    MEMORY_FILE = 'global_memory/memory.json'
    SERIES_FILE = 'chat_series/chat_series.json'

    @staticmethod
    def update_config_voice():
        Config.config['polly']['LanguageCode'] = Config.voiceLanguageList[Config.voiceIndex]
        Config.config['polly']['VoiceId'] = Config.voiceNameList[Config.voiceIndex]

# 全局路由处理
class URL:
    # 创建首页路由
    @staticmethod
    @app.route("/")
    def index():
        return render_template("index.html")

    # 创建prism库的路由
    @staticmethod
    @app.route('/prism/<filename>')
    def prism(filename):
        return send_from_directory('prism', filename)  # 直接从prism文件夹发送文件

# 模型输出处理
class Bedrock:
    interrupted = False
    audio_cache = dict()  # 存储音频数据

    # 输入
    @staticmethod
    def generate_response(user_text, user_text_with_prompt):
        print(Config.audio)
        ai_response_html = ''

        print(Config.audio)
        socketio.emit("begin_generate_response")
        ai_response = ''
        raw_ai_response = ''
        ai_response_char_iter = Bedrock.char(user_text_with_prompt)
        for char in ai_response_char_iter:

            if Bedrock.interrupted:
                break
            ai_response += char
            raw_ai_response += char
            if char == '\\':
                ai_response += char
            ai_response_html = markdown.markdown(ai_response, extensions=['fenced_code', 'extra'])
            socketio.emit("refresh_ai_response", ai_response_html)
        if Config.audio:
            UserInput.history.append('用户（语音）：' + user_text)
        else:
            UserInput.history.append('用户（文字）：' + user_text)
        if Bedrock.interrupted:
            for _ in ai_response_char_iter:
                pass
            Bedrock.interrupted = False
            if Config.audio:
                UserInput.history.append('AI回复（语音，输出被打断）：' + raw_ai_response)
            else:
                UserInput.history.append('AI回复（文字，输出被打断）：' + raw_ai_response)
        else:
            if Config.audio:
                UserInput.history.append('AI回复（语音）：' + raw_ai_response)
            else:
                UserInput.history.append('AI回复（文字）：' + raw_ai_response)
        if Config.audio:
            audio_url = Bedrock.bedrock_text_to_speech(raw_ai_response)
            socketio.emit("ai_response_audio",
                          audio_url)
        socketio.emit("ai_response_end", raw_ai_response)

        GlobalHistory.global_history.add_entry(user_text, raw_ai_response)

        ChatSeries.chat_series[ChatSeries.current_chat_index].update_chat_history(
            {"ai": raw_ai_response, "ai_html": ai_response_html})  # 更新聊天记录

        # 定义正则表达式（贪婪模式匹配）
        pattern = r'<global-memory>(.*?)</global-memory>'
        memory_matches = re.findall(pattern, raw_ai_response, re.DOTALL)
        if memory_matches:
            print(memory_matches)
            for content in enumerate(memory_matches):
                GlobalMemory.add_memory(content[-1])

        # 在生成完响应后，重新启用开始按钮
        socketio.emit("toggle_button", {"button_id": "start-button", "state": False})  # 让开始按钮工作

    # 调用 Amazon Bedrock GPT 处理对话
    @staticmethod
    def char(user_text):
        # noinspection PyShadowingNames
        try:
            body = UserInput.define_body(user_text, MultiModal.decode_images(), MultiModal.read_txt())
            body_json = json.dumps(body)
            response = bedrock_runtime.invoke_model_with_response_stream(
                body=body_json,
                modelId=Config.config['bedrock']['api_request']['modelId'],
                accept=Config.config['bedrock']['api_request']['accept'],
                contentType=Config.config['bedrock']['api_request']['contentType']
            )
            print(Config.config['bedrock']['api_request']['modelId'])

            bedrock_stream = response.get('body')

            model_provider = Config.model_id.split('.')[0]
            text = ''

            if bedrock_stream:
                for event in bedrock_stream:
                    if Bedrock.interrupted:
                        time.sleep(1)
                        return
                    chunk = event.get('chunk')
                    if chunk:
                        text = ''
                        if model_provider == 'meta':
                            chunk_obj = json.loads(chunk.get('bytes').decode())
                            text = chunk_obj['generation']
                        elif model_provider == 'anthropic':
                            if "claude-3" in Config.model_id:
                                chunk_obj = json.loads(chunk.get('bytes').decode())
                                if chunk_obj['type'] == 'message_delta':
                                    if not chunk_obj['delta']['stop_sequence']:
                                        chunk_obj['delta']['stop_sequence'] = "none"
                                    socketio.emit("retrievedInfo",{
                                        "stopReason": chunk_obj['delta']['stop_reason'],
                                        "stopSequence": chunk_obj['delta']['stop_sequence'],
                                        "outputTokens": chunk_obj['usage']['output_tokens']
                                    })

                                    ChatSeries.chat_series[ChatSeries.current_chat_index].update_chat_history(
                                        {"stopReason": chunk_obj['delta']['stop_reason'],
                                        "stopSequence": chunk_obj['delta']['stop_sequence'],
                                        "outputTokens": chunk_obj['usage']['output_tokens']})  # 更新聊天记录

                                    print(f"\nStop reason: {chunk_obj['delta']['stop_reason']}")
                                    print(f"Stop sequence: {chunk_obj['delta']['stop_sequence']}")
                                    print(f"Output tokens: {chunk_obj['usage']['output_tokens']}")
                                if chunk_obj['type'] == 'content_block_delta':
                                    if chunk_obj['delta']['type'] == 'text_delta':
                                        text = chunk_obj['delta']['text']

                            else:
                                chunk_obj = json.loads(chunk.get('bytes').decode())
                                text = chunk_obj['completion']
                        else:
                            raise NotImplementedError('Unknown model provider.')
                        for char in text:
                                yield char
                        text = ''


        except Exception as e:
            e_text = f"API 调用出错: {e}"
            print(f"API 调用出错: {e}")
            for char in e_text:
                yield char

    # 添加音频专属路由处理
    @staticmethod
    @app.route('/stream-audio/<audio_id>')
    def stream_audio(audio_id):
        audio_bytes = Bedrock.audio_cache.get(audio_id)
        if audio_bytes:
            audio_stream = io.BytesIO(audio_bytes)
            response = send_file(
                audio_stream,
                mimetype='audio/mpeg',
                as_attachment=False,
                download_name='speech.mp3'
            )
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Accept-Ranges'] = 'bytes'
            return response
        else:
            return "Audio expired", 404

    @staticmethod
    # 音频输出函数（音频传给html）
    def bedrock_text_to_speech(text):
        # noinspection PyShadowingNames
        try:
            polly = boto3.client('polly', region_name=Config.config['region'])
            polly_response = polly.synthesize_speech(
                Text=text,
                Engine=Config.config['polly']['Engine'],
                LanguageCode=Config.config['polly']['LanguageCode'],
                VoiceId=Config.config['polly']['VoiceId'],
                OutputFormat=Config.config['polly']['OutputFormat'],
            )

            audio_bytes = polly_response['AudioStream'].read()  # 🧠 一次性读完
            audio_id = str(uuid.uuid4())
            Bedrock.audio_cache[audio_id] = audio_bytes  # 💾 存储为字节
            return f'/stream-audio/{audio_id}'  # 🔊 返回唯一标识
        except Exception as e:
            print(f"API 调用出错: {e}")
            return "AI 处理请求时发生错误"

    # 响应：打断输出
    @staticmethod
    @socketio.on("interrupt")
    def interrupt():
        Bedrock.interrupted = True
        print('打断输入')

# 设置窗口
class Settings:
    temperature = Config.temperature
    audio = Config.audio
    use_RAG = Config.use_RAG
    language = None
    embedding_model_name = Config.embedding_model_name_index
    settings = None
    file_path = Config.SETTINGS_FILE

    @staticmethod
    @socketio.on('start_settings_init')
    def settings_init():
        """
            安全加载和解析 JSON 配置文件
            返回包含所有配置的字典，并自动转换数据类型
            """
        try:
            config_file = Path(Settings.file_path)

            # 🛡️ 检查文件是否存在
            if not config_file.is_file():
                raise FileNotFoundError(f"❌ 配置文件 {Settings.file_path} 不存在")

            # 📖 读取文件内容
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            Config.temperature = float(config_data.get('temperature', Config.temperature))
            Config.audio = bool(config_data.get('audio', Config.audio))
            Config.voiceIndex = int(config_data.get('language', Config.voiceIndex))
            Config.embedding_model_name_index = int(config_data.get('embedding_model', Config.embedding_model_name_index))
            Config.use_RAG = int(config_data.get('use_RAG', Config.use_RAG))

            print('settings_init:', Config.temperature, Config.audio, Config.voiceIndex, Config.use_RAG, Config.embedding_model_name_index)

            socketio.emit("settings_init", {
                "temperature": Config.temperature,
                "audio": Config.audio,
                "language": Config.voiceIndex,
                "use_RAG": Config.use_RAG,
                "embedding_model": Config.embedding_model_name_index})

        except json.JSONDecodeError:
            print("❌ JSON 格式错误，请检查配置文件语法")
        except Exception as e:
            print(f"❌ 读取配置出错: {str(e)}")

    # 响应：加载设置
    @staticmethod
    @socketio.on('update_settings')
    def handle_update_settings(data):
        Settings.temperature = data.get('temperature')
        Settings.audio = bool(data.get('audio_state'))
        Settings.language = data.get('language')
        Settings.use_RAG = data.get('use_RAG')
        Settings.embedding_model_name = data.get('embedding_model')

        # 加载当前设置
        Settings.settings = Settings.load_settings()
        if Settings.temperature is not None:
            Settings.settings['temperature'] = Settings.temperature
        if Settings.audio is not None:
            Settings.settings['audio'] = Settings.audio
            Config.audio = Settings.settings['audio']
            print(Config.audio)
        if Settings.language is not None:
            Settings.settings['language'] = Settings.language
            Config.voiceIndex = int(Settings.settings['language'])
            Config.update_config_voice()
        if Settings.use_RAG is not None:
            Settings.settings['use_RAG'] = Settings.use_RAG
            Config.use_RAG = Settings.settings['use_RAG']
        if Settings.embedding_model_name is not None:
            Settings.settings['embedding_model'] = Settings.embedding_model_name
            Config.embedding_model_name_index = int(Settings.settings['embedding_model'])

        # 保存设置
        Settings.save_settings(Settings.settings)

        # 返回响应
        socketio.emit('settings-saved', {'success': True, 'settings': Settings.settings})
        print('Updated settings:', Settings.settings)

    @staticmethod
    # 保存设置文件
    def save_settings(settings):

        with open(Config.SETTINGS_FILE, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)  # 排他锁
            json.dump(settings, f, indent=4)
            portalocker.lock(f, portalocker.LOCK_UN)  # 释放锁

    @staticmethod
    # 加载设置
    def load_settings():
        with open(Config.SETTINGS_FILE, 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)  # 共享锁
            settings = json.load(f)
            portalocker.lock(f, portalocker.LOCK_UN)  # 释放锁
        return settings

# 用于暂停输入
class UserInputAudio:
    shutdown_executor = False
    executor = None

    @staticmethod
    def set_executor(executor):
        UserInputAudio.executor = executor

    @staticmethod
    def start_shutdown_executor():
        UserInputAudio.shutdown_executor = True

    @staticmethod
    def start_user_input_loop():
        pass

    @staticmethod
    def is_executor_set():
        return UserInputAudio.executor is not None

    @staticmethod
    def is_shutdown_scheduled():
        return UserInputAudio.shutdown_executor

# 语音转文本，传递给Bedrock模型生成回复
class EventHandler(TranscriptResultStreamHandler):
    text = []
    last_time = 0
    sample_count = 0
    max_sample_counter = 4
    history = []

    def __init__(self, transcript_result_stream: TranscriptResultStream, bedrock_wrapper, loop):
        super().__init__(transcript_result_stream)
        self.bedrock_wrapper = bedrock_wrapper
        self.loop = loop

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        if UserInputAudio.shutdown_executor:
            return
        if not self.bedrock_wrapper.is_speaking():
            if results:
                for result in results:
                    EventHandler.sample_count = 0
                    if not result.is_partial:
                        for alt in result.alternatives:
                            print(alt.transcript, flush=True, end=' ')
                            EventHandler.text.append(alt.transcript)

            else:
                EventHandler.sample_count += 1
                if EventHandler.sample_count == EventHandler.max_sample_counter:
                    if len(EventHandler.text) != 0:
                        if UserInputAudio.shutdown_executor:
                            return
                        input_text = ''.join(EventHandler.text)
                        executor = ThreadPoolExecutor(max_workers=1)
                        UserInputAudio.set_executor(executor)
                        self.loop.run_in_executor(
                            executor,
                            self.bedrock_wrapper.recognition_loop,
                            input_text
                        )

                    EventHandler.text.clear()
                    EventHandler.sample_count = 0

# 从麦克风捕获音频流，发送到Amazon Transcribe转录
class MicStream:
    transcribe_streaming = TranscribeStreamingClient(region=Config.config['region'])

    async def mic_stream(self):
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue()

        def callback(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

        stream = sounddevice.RawInputStream(
            channels=1, samplerate=16000, callback=callback, blocksize=2048 * 2, dtype="int16")
        with stream:
            while True:
                indata, status = await input_queue.get()
                yield indata, status

    async def write_chunks(self, stream):
        async for chunk, status in self.mic_stream():
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
        await stream.input_stream.end_stream()

    async def basic_transcribe(self, loop):
        loop.run_in_executor(ThreadPoolExecutor(max_workers=1), UserInputAudio.start_user_input_loop)
        if UserInputAudio.is_shutdown_scheduled():
            return
        stream = await MicStream.transcribe_streaming.start_stream_transcription(
                language_code=Config.voiceList[Config.voiceIndex],
                media_sample_rate_hz=16000,
                media_encoding="pcm",
        )
        handler = EventHandler(stream.output_stream, UserInput.Speech, loop)
        await asyncio.gather(self.write_chunks(stream), handler.handle_events())

# 用户输入处理
class UserInput:
    history = []

    @staticmethod
    @socketio.on('current_model_index')
    def current_model_index(current_model_index):
        model_index = int(current_model_index)
        Config.model_index = model_index
        Config.model_id = os.getenv('MODEL_ID', Config.model_id_list[model_index])
        Config.config['bedrock']['api_request'] = api_request_list[Config.model_id]
        print(f'获取更新模型索引。Config.config[api_request]：{api_request_list[Config.model_id]}')

    # 发送给模型的请求体
    @staticmethod
    def define_body(text, image_list=None, txt_from_pdf=None):

        model_id = Config.config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]
        body = Config.config['bedrock']['api_request']['body']

        if txt_from_pdf:
            print(f"读取上传的pdf；{txt_from_pdf}")
            text = f"用户上传的pdf内容：{txt_from_pdf}, 用户：" + text

        if model_provider == 'amazon':
            body['inputText'] = text
        elif model_provider == 'meta':
            if 'llama3' in model_id:
                with open('./llama_prompt', 'r', encoding='utf-8') as file:
                    prompt = file.read()
                body['prompt'] = prompt.format(text=text)
            else:
                body['prompt'] = f"<s>[INST] {text}, please output in Chinese. [/INST]"
        elif model_provider == 'anthropic':
            if "claude-3" in model_id:
                s = Settings.load_settings()
                body['temperature'] = float(s['temperature'])
                import claude3_prompts as cp
                if image_list:
                    add = [
                        {
                            "role": "user",
                            "content": [
                                *[
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": image['type'],
                                            "data": image['data'],
                                        },
                                    }
                                    for image in image_list
                                ],
                                {
                                    "type": "text",
                                    "text": text,
                                },
                            ],
                        }
                    ]
                else:
                    add = [
                        {"role": "user", "content": text}
                    ]
                body['messages'] = cp.promptsMessages + add
                if Config.audio:
                    body['system'] = cp.promptsSystemAudio
                else:
                    body['system'] = cp.promptsSystemText

            else:
                body['prompt'] = f'\n\nHuman: {text}\n\nAssistant:'
        elif model_provider == 'cohere':
            body['prompt'] = text
        elif model_provider == 'mistral':
            body['prompt'] = f"<s>[INST] {text}, please output in Chinese. [/INST]"
        else:
            raise Exception('Unknown model provider.')

        return body

    class Speech:
        speaking = False
        loop = None

        @staticmethod
        def cancel_all_tasks():
            # 获取所有未完成的任务
            tasks = [task for task in asyncio.all_tasks(UserInput.Speech.loop) if not task.done()]

            # 取消所有任务
            for task in tasks:
                task.cancel()

            UserInput.Speech.loop.run_until_complete(asyncio.wait(tasks))

            UserInputAudio.executor.shutdown(wait=False)
            UserInputAudio.shutdown_executor = False
            UserInputAudio.executor = None

            UserInput.Speech.loop.stop()
            UserInput.Speech.loop.close()
            asyncio.set_event_loop(None)

        @staticmethod
        def is_speaking():
            return UserInput.Speech.speaking

        @staticmethod
        # 识别到音频并传递给模型与前端
        def recognition_loop(user_text):
            user_text = "<p>" + user_text + "</p>"
            UserInput.Speech.speaking = True
            UserInputAudio.shutdown_executor = True
            # 处理用户输入
            socketio.emit("recognized_text", user_text)  # 发送用户输入到前端
            ChatSeries.chat_series[ChatSeries.current_chat_index].update_chat_history({"user": user_text})
            if len(UserInput.history) > 20:
                UserInput.history = UserInput.history[-20:]
            text_history = '\n'.join(UserInput.history)
            memories = GlobalMemory.get_memories()
            if Config.use_RAG:
                retrieved_text = RAG.rag.get_retrieved_text(user_text)
                user_text_with_prompt = f'''<prompt>
以下是对话记录（最后的是最新一次的对话记录）：<history>
{text_history}
</history>
以下是全局对话记录（包含了所有对话中的最近十条消息）：<global_history>
{GlobalHistory.global_history._history}
</global_history>
这是已经添加的关于用户的一些全局记忆：<global-memory>
{memories}
</global-memory>
用用户的问题可以检索到以下相关信息：
<rag>
{retrieved_text}
</rag>
用户的对话最可能与最新的一次（历史中最后一个）对话有关。
这是用户输入，请回答：<question>
{user_text}
</question>
你的所有输出所用的语言必须与question内的语言一致。
</prompt>
'''
            else:
                user_text_with_prompt = f'''<prompt>
以下是对话记录（最后的是最新一次的对话记录）：<history>
{text_history}
</history>
以下是全局对话记录（包含了所有对话中的最近十条消息）：<global_history>
{GlobalHistory.global_history._history}
</global_history>
这是已经添加的关于用户的一些全局记忆：<global-memory>
{memories}
</global-memory>
用户的对话最可能与最新的一次（历史中最后一个）对话有关。
这是用户输入，请回答：<question>
{user_text}
</question>
你的所有输出所用的语言必须与question内的语言一致。
</prompt>
'''

            print(user_text_with_prompt)
            Bedrock.generate_response(user_text, user_text_with_prompt)  # 生成音频

            # 语音识别结束后，重新启用开始按钮
            socketio.emit("toggle_button", {"button_id": "start-button", "state": False})  # 让开始按钮工作

        # 处理客户端语音识别请求
        @staticmethod
        @socketio.on("start_recognition")
        def handle_recognition():
            print('语音识别已开始')
            if not UserInput.Speech.speaking:
                socketio.emit("toggle_button", {"button_id": "start-button", "state": True})  # 禁用开始按钮
                UserInput.Speech.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(UserInput.Speech.loop)
                try:
                    UserInputAudio.shutdown_executor = False
                    UserInput.Speech.loop.run_until_complete(MicStream().basic_transcribe(UserInput.Speech.loop))
                except (KeyboardInterrupt, Exception) as e:
                    print(e)
                print('语音识别已结束yyeyeyeyeyeye')
                UserInput.Speech.cancel_all_tasks()

        @staticmethod
        @socketio.on("audio_completed")
        def audio_completed():
            if UserInput.Speech.speaking:
                print('语音识别已结束')
                socketio.emit("toggle_button", {"button_id": "start-button", "state": False})
                UserInputAudio.start_shutdown_executor()
                UserInput.Speech.speaking = False

    # 处理客户端文字识别请求
    @staticmethod
    @socketio.on("user_text")
    def user_text_loop(user_text):
        print(f"收到用户输入: {user_text}")
        user_text_html = markdown.markdown(user_text.replace('\n', '\n\n'), extensions=['fenced_code', 'extra'])
        socketio.emit("recognized_text", user_text_html)  # 发送用户输入到前端
        ChatSeries.chat_series[ChatSeries.current_chat_index].update_chat_history({"user": user_text_html})  # 更新聊天记录

        UserInput.history.append(user_text)
        if len(UserInput.history) > 10:
            UserInput.history = UserInput.history[:10]
        text_history = '\n'.join(UserInput.history)
        memories = GlobalMemory.get_memories()
        if Config.use_RAG:
            retrieved_text = RAG.rag.get_retrieved_text(user_text)
            user_text_with_prompt = f'''<prompt>
以下是对话记录（最后的是最新一次的对话记录）：<history>
{text_history}
</history>
以下是全局对话记录（包含了所有对话中的最近十条消息）：<global_history>
{GlobalHistory.global_history._history}
</global_history>
这是已经添加的关于用户的一些全局记忆：<global-memory>
{memories}
</global-memory>
用用户的问题可以检索到以下相关信息：
<rag>
{retrieved_text}
</rag>
用户的对话最可能与最新的一次（历史中最后一个）对话有关。
这是用户输入，请回答：<question>
{user_text}
</question>
你的所有输出所用的语言必须与question内的语言一致。
</prompt>
'''
        else:
            user_text_with_prompt = f'''<prompt>
以下是对话记录（最后的是最新一次的对话记录）：<history>
{text_history}
</history>
以下是全局对话记录（包含了所有对话中的最近十条消息）：<global_history>
{GlobalHistory.global_history._history}
</global_history>
这是已经添加的关于用户的一些全局记忆：<global-memory>
{memories}
</global-memory>
用户的对话最可能与最新的一次（历史中最后一个）对话有关。
这是用户输入，请回答：<question>
{user_text}
</question>
你的所有输出所用的语言必须与question内的语言一致。
</prompt>
'''
        print(user_text_with_prompt)

        Bedrock.generate_response(user_text, user_text_with_prompt)  # 生成音频

    @staticmethod
    def update_history():
        current_chat = ChatSeries.chat_series[ChatSeries.current_chat_index]
        if current_chat.chat_history:
            for message in current_chat.chat_history:
                UserInput.history.append("用户：" + message["user"])
                try:
                    UserInput.history.append("模型回复：" + message["ai"])
                except KeyError:
                    UserInput.history.append("模型回复：" + "输出被打断")
            if len(UserInput.history) > 20:
                UserInput.history = UserInput.history[-20:]
        else:
            UserInput.history = []

# 跨对话记忆管理
class GlobalHistory:
    global_history = None

    """
    一个用于LLM对话的固定大小历史管理器，支持JSON持久化。
    最多存储 `max_size` 条问答对，并将其保存在本地文件中。
    """
    def __init__(self, file_path: Optional[str] = None, max_size: int = 10):
        self.max_size = max_size
        # 确定JSON文件的路径
        self.file_path = Path(file_path or r'global_memory/GlobalHistory.json')
        # 加载已有历史或初始化为空列表
        self._history: List[Tuple[str, str]] = self._load()

    def add_entry(self, question: str, response: str) -> None:
        """
        添加一条新的问答对，若超出最大容量则删除最早的条目，
        并立即将最新历史持久化到JSON文件。
        """
        self._history.append((question, response))
        # 若超过最大容量，则删除最早的多余条目
        if len(self._history) > self.max_size:
            excess = len(self._history) - self.max_size
            self._history = self._history[excess:]
        self._save()

    def get_history(self) -> List[Tuple[str, str]]:#返回当前的历史列表
        return list(self._history)

    def clear(self) -> None:#清空内存中的历史，并删除JSON文件。

        self._history.clear()
        if self.file_path.exists():
            self.file_path.unlink()

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        # 返回对象的字符串表示，包含当前大小和文件路径
        return f"<HistoryManager size={len(self._history)}/{self.max_size} file='{self.file_path}'>"

    def _load(self) -> List[Tuple[str, str]]:
        """
        从JSON文件加载历史，返回问答元组列表。
        如果文件不存在或内容无效，则返回空列表。
        """
        if not self.file_path.exists():
            return []
        try:
            data = json.loads(self.file_path.read_text(encoding='utf-8'))
            # 期望文件内容为包含 'question' 和 'response' 的字典列表
            history = [(item['question'], item['response']) for item in data]
            # 若文件中记录超过最大容量，则只保留最新的部分
            return history[-self.max_size:]
        except (json.JSONDecodeError, KeyError, TypeError):
            # 返回空列表以防解析错误
            return []

    def _save(self) -> None:
        """
        将当前历史持久化写入JSON文件。
        """
        data = [{'question': q, 'response': r} for q, r in self._history]
        self.file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

    @classmethod
    def load_from_file(cls, file_path: str, max_size: int = 10) -> 'GlobalHistory':
        """
        便捷构造方法：根据已有文件路径创建HistoryManager实例。
        """
        return cls(file_path=file_path, max_size=max_size)

# 多模态
class MultiModal:
    doc_list = []
    using_doc_list = []

    @staticmethod
    @socketio.on("clear_uploads")
    def clear_uploads():
        MultiModal.doc_list = []
        MultiModal.using_doc_list = []
        print('清空上传列表')

        """
            清空 uploads 目录下的所有文件和子目录
            保留 uploads 目录本身
            """
        uploads_dir = Path('./uploads')

        try:
            # 🛡️ 检查目录是否存在
            if not uploads_dir.exists():
                print(f"⚠️ 目录不存在: {uploads_dir}")
                return False

            # 🗑️ 遍历并删除所有内容
            for item in uploads_dir.iterdir():
                try:
                    if item.is_file():
                        # 📃 删除文件
                        item.unlink()
                        print(f"✅ 已删除文件: {item.name}")
                    else:
                        # 📂 删除子目录
                        shutil.rmtree(item)
                        print(f"✅ 已删除目录: {item.name}")
                except Exception as e:
                    print(f"❌ 删除失败 {item.name}: {str(e)}")

            print(f"🎉 成功清空 {uploads_dir} 目录")
            return True

        except PermissionError:
            print(f"❌ 权限不足，无法访问目录: {uploads_dir}")
        except Exception as e:
            print(f"❌ 操作失败: {str(e)}")
        return False

    # 安全校验文件类型
    @staticmethod
    def allowed_file(filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    # 生成安全文件名
    @staticmethod
    def secure_filename_with_timestamp(filename):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        print(f"生成的文件名：{timestamp}_{filename}")
        base_name = secure_filename(filename)
        return f"{timestamp}_{base_name}"

    # 文件上传路由
    @staticmethod
    @app.route('/upload', methods=['POST'])
    def handle_upload():
        # 检查请求中是否有文件
        if 'file' not in request.files:
            return jsonify(error="没有选择文件"), 400

        file = request.files['file']

        # 双重安全检查
        if file.filename == '':
            return jsonify(error="文件名不能为空"), 400
        if not MultiModal.allowed_file(file.filename):
            return jsonify(error="不支持的文件类型"), 415

        try:
            # 🔒 生成安全文件名
            safe_filename = MultiModal.secure_filename_with_timestamp(file.filename)

            # 📂 自动创建上传目录
            save_path = os.path.join(app.config['UPLOAD_FOLDER'])
            os.makedirs(save_path, exist_ok=True)

            # 💾 保存文件到本地
            file.save(os.path.join(save_path, safe_filename))
            MultiModal.doc_list.append(safe_filename)
            MultiModal.using_doc_list.append(safe_filename)
            if len(MultiModal.using_doc_list) > 5:
                MultiModal.using_doc_list.pop(0)

            socketio.emit("upload_files", {"doc_list": MultiModal.doc_list, "using_doc_list": MultiModal.using_doc_list})

            return jsonify(
                success=True,
                message="文件上传成功！",
                filename=safe_filename,
                save_path=os.path.abspath(save_path)
            ), 200

        except Exception as e:
            return jsonify(error=f"上传失败: {str(e)}"), 500

    @staticmethod
    @socketio.on("remove_file")
    def remove_file(index):
        index = int(index)
        file_to_remove = MultiModal.doc_list[index]
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], file_to_remove)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)
            print("文件已删除")

        MultiModal.doc_list.pop(index)
        if len(MultiModal.doc_list) >= 5:
            MultiModal.using_doc_list = MultiModal.doc_list[:5]
            print(MultiModal.using_doc_list)
        else:
            MultiModal.using_doc_list.pop(index)
            print(MultiModal.using_doc_list)

        socketio.emit("upload_files", {"doc_list": MultiModal.doc_list, "using_doc_list": MultiModal.using_doc_list})

    @staticmethod
    def decode_images():
        image_list = []
        using_image_list = [filename for filename in MultiModal.using_doc_list if filename[-4:] != '.pdf']

        for doc in using_image_list:
            with open(os.path.join(app.config['UPLOAD_FOLDER'], doc), 'rb') as f:
                image = dict()
                image['data'] = base64.b64encode(f.read()).decode("utf-8")
                image['type'] = "image/" + doc.split('.')[-1]
                image_list.append(image)

        return image_list

    @staticmethod
    def read_txt():
        txt_list = []
        using_txt_list = [filename for filename in MultiModal.using_doc_list if filename[-4:] == '.pdf']
        for doc in using_txt_list:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], doc)
            txt_list.append(RAG.process_pdf(file_path))
        return '\n'.join(txt_list)

# RAG
class RAG:
    doc_name = []
    doc_texts = []
    doc_indexes = [] # 每一个文档文本结束的索引
    rag = None
    raw_pdf_list = []

    @staticmethod
    @socketio.on("RAG_refresh")
    def refresh_RAG():
        if Config.use_RAG and ((not RAG.rag) or (RAG.rag.embedding_model_name != Config.embedding_model_names[Config.embedding_model_name_index])):
            Config.use_RAG = False
            RAG.rag = RAG()
            print(Config.embedding_model_names[Config.embedding_model_name_index])
            print('RAG重新加载')
        elif Config.use_RAG:
            print('RAG已就绪')
            socketio.emit("RAG_loaded")
        else:
            socketio.emit("RAG_unavailable")

    @staticmethod
    def get_embeddings(model, tokenizer, texts, batch_size=32):
        # 批量处理避免内存溢出
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )

            with torch.no_grad():
                outputs = model(**inputs)

            # 使用最后一层隐藏状态的均值作为嵌入
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings)

        return torch.cat(embeddings, dim=0).numpy().astype('float32')

    # 加载模型
    def __init__(self):
        self.embedding_model_name = Config.embedding_model_names[Config.embedding_model_name_index]
        self.embedding_model = None
        self.embeddings = None

        def load_embeddings():
            try:
                socketio.emit("RAG_loading")
                import os
                os.environ["USE_FLASH_ATTENTION"] = "0"
                print(f"Loading {self.embedding_model_name}")
                if self.embedding_model_name == "Alibaba-NLP/get-multilingual-base":
                    self.embedding_model = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base")
                    self.tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base")

                    self.embeddings = RAG.get_embeddings(self.embedding_model, self.tokenizer, RAG.doc_texts)
                else:
                    self.embedding_model = SentenceTransformer(self.embedding_model_name, device="cuda", trust_remote_code=True)
                    self.embeddings = self.embedding_model.encode(RAG.doc_texts, normalize_embeddings=True, batch_size=32)
                print("Embeddings loaded successfully.")
                Config.use_RAG = True
                socketio.emit("RAG_loaded")
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                socketio.emit("RAG_loading_failed", str(e).split(":")[0])

        embedding_thread = threading.Thread(target=load_embeddings)
        embedding_thread.daemon = True  # 设置为守护线程（主线程退出时自动关闭）
        embedding_thread.start()

    # 文本初始化
    @staticmethod
    def text_init():
        folder = Path(r"RAG/processed_txt/")  # 替换实际路径
        embed_model = None
        semantic_chunker = None
        for f in folder.iterdir():
            if not f.is_file():
                continue
            txt = ''
            output_file = os.path.join(r"RAG\chunked_txt", f.name)
            if os.path.exists(output_file):
                print(f"{output_file} 已存在，跳过分块处理。")
                RAG.doc_name.append(f.name)
                with open(output_file, "r", encoding='utf-8') as file:
                    txt = file.read()
                    RAG.doc_texts += txt.split('bubu')
                continue
            else:
                if not embed_model:
                    embed_model = HuggingFaceEmbeddings(
                        model_name="BAAI/bge-base-en-v1.5",
                        model_kwargs={'device': 'cuda'},  # 指定使用CUDA
                        encode_kwargs={'normalize_embeddings': True}
                    )
                if not semantic_chunker:
                    semantic_chunker = SemanticChunker(
                        embeddings=embed_model,
                        breakpoint_threshold_type="interquartile"
                    )
                RAG.doc_name.append(f.name)
                with open(r"RAG/processed_txt/" + f.name, 'r', encoding='utf-8') as file:
                    loader = TextLoader(r"RAG/processed_txt/" + f.name, encoding="utf-8")
                    documents = loader.load()
                    texts = [doc.page_content for doc in documents]
                    semantic_chunks = semantic_chunker.create_documents(texts)  # ✅ 传入纯文本列表
                    for chunk in semantic_chunks:
                        RAG.doc_texts.append(chunk.page_content)
                        txt += chunk.page_content + 'bubu'
                with open(output_file, 'w', encoding='utf-8') as file:
                    file.write(txt) # 保存分块后的文本
                print(f"{output_file} 已完成分块处理。")
            RAG.doc_indexes.append(len(RAG.doc_texts) - 1)

    @staticmethod
    def process_pdf(filename):

        # 添加环境变量
        import os
        poppler_path = r'./pdf_dependencies/poppler-24.08.0/Library/bin'
        os.environ["PATH"] += os.pathsep + poppler_path
        pytesseract_path = r'./pdf_dependencies/Tesseract-OCR'
        os.environ["PATH"] += os.pathsep + pytesseract_path
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            filename=filename,
            infer_table_structure=True,  # infer_table_structure=True 自动选择 hi_res 策略
            include_page_breaks=True
        )

        # 按 PageBreak 分页
        all_sublists = []
        temp = []
        for el in elements:
            if el.category == "PageBreak":
                if temp:
                    all_sublists.append(temp)
                    temp = []
            else:
                temp.append(el)
        if temp:
            all_sublists.append(temp)

        pdf_elements = []
        # 处理每一页
        for sublist in all_sublists:
            # 提取所有元素的坐标
            coordinates = [element.metadata.coordinates.to_dict() for element in sublist]
            all_points = [coordinate['points'] for coordinate in coordinates]

            # 计算中线
            top_left_min = min(points[0][0] for points in all_points)  # 最小的 x 坐标
            bottom_right_max = max(points[2][0] for points in all_points)  # 最大的 x 坐标
            mid_line_x_coordinate = (top_left_min + bottom_right_max) / 2
            print(f"最小横坐标：{top_left_min}, 最大横坐标：{bottom_right_max}, 中线：{mid_line_x_coordinate}")

            # 分栏
            left_column = []
            right_column = []
            for element, points in zip(sublist, all_points):
                top_left = min(points, key=lambda p: (p[0], p[1]))  # 左上角点
                if top_left[0] < mid_line_x_coordinate:
                    left_column.append(element)
                else:
                    right_column.append(element)

            # 按 y 坐标排序
            left_column_sorted = sorted(left_column, key=lambda element:
            min(element.metadata.coordinates.to_dict()['points'], key=lambda p: (p[0], p[1]))[1])
            right_column_sorted = sorted(right_column, key=lambda element:
            min(element.metadata.coordinates.to_dict()['points'], key=lambda p: (p[0], p[1]))[1])

            # 打印结果
            print("\n左栏元素：")
            for element in left_column_sorted:
                print(element)

            print("\n右栏元素：")
            for element in right_column_sorted:
                print(element)

            print("\n" + "=" * 50 + "\n")  # 添加分隔符

            page_elements = [str(element) for element in left_column_sorted] + [str(element) for element in right_column_sorted]
            s_page_elements = '\n'.join(page_elements)
            pdf_elements.append(s_page_elements)

        return '\n'.join(pdf_elements)

    @staticmethod
    def process_pdf_for_rag():
        folder = Path(r"RAG/raw_pdf/")
        for f in folder.iterdir():
            if not f.is_file():
                continue
            RAG.raw_pdf_list.append(f.name)

        import os

        for file_name in RAG.raw_pdf_list:
            input_file = os.path.join("RAG/raw_pdf", file_name)
            output_file = os.path.join("RAG/processed_txt", os.path.splitext(file_name)[0] + ".txt")

            # 判断输出文件是否已存在
            if os.path.exists(output_file):
                print(f"{output_file} 已存在，跳过处理。")
            else:
                pdf_result = RAG.process_pdf(input_file)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(pdf_result)
                print(f"{file_name} 已处理完毕，结果保存到 {output_file}。")

    # 根据用户输入寻找关联
    def get_retrieved_text(self, input_text):
        retrieved_text = dict()
        max_indexes = []
        max_index = 0
        pieced = False
        retrieved_texts = []

        def find_article_name(index):
            for i in range(len(RAG.doc_indexes)):
                if index <= RAG.doc_indexes[i]:
                    return RAG.doc_name[i]
            return None

        if pieced:
            pieces = 3
            while (len(input_text) - 6) % pieces != 0:
                input_text += ' '
            input_text += ' '
            input_text_pieces = [input_text[pieces * i: pieces * (i + 1) + 6] for i in range((len(input_text) - 6) // pieces)]
            input_text_pieces.append(input_text)
            input_embeddings = [self.embedding_model.encode(input_text_piece, normalize_embeddings=True) for input_text_piece in input_text_pieces]
            scores_list = [self.embedding_model.similarity(input_embedding, self.embeddings) for input_embedding in input_embeddings]
            max_indexes = [int(np.argmax(scores)) for scores in scores_list]
            max_indexes = list(set(max_indexes))  # 去重

            for index in max_indexes:
                retrieved_text["article"] = find_article_name(index)
                retrieved_text["content"] = RAG.doc_texts[index]

                retrieved_texts.append(retrieved_text)

            return retrieved_texts


        else:

            input_text_iter = Bedrock.char(
                "用户想使用RAG检索信息。请你使用HyDE技术，理解用户需求并生成一个理想的答案，你的回答将用于RAG检索，请不要添油加醋。以下是用户的问题：" + input_text)

            regenerated_rag_questions = ''
            for iter in input_text_iter:
                regenerated_rag_questions += iter
            regenerated_rag_questions += "用户问题：" + input_text + "ai理解"
            print(f"用户提问+ai重新阐述：{regenerated_rag_questions}")

            if self.embedding_model_name == 'Alibaba-NLP/get-multilingual-base':
                input_embedding = np.array(self.embedding_model.encode([regenerated_rag_questions]), dtype='float32')
                scores = cosine_similarity(input_embedding, self.embeddings)[0]  # 降维
            else:
                input_embedding = self.embedding_model.encode([regenerated_rag_questions], normalize_embeddings=True)
                scores = self.embedding_model.similarity(input_embedding, self.embeddings)

            # 将 scores 转换为一维数组
            scores = scores.squeeze()  # 如果 scores 是二维张量，例如形状为 (1, n)，则将其转换为一维数组
            if scores.ndim != 1:
                raise ValueError("scores 必须是一维数组")
            if len(scores) < 3:
                raise ValueError("scores 中元素不足，无法获取前三名")

            # 获取得分最高的前三个索引（使用 PyTorch 的 argsort）
            import torch
            top_3_indices = torch.argsort(scores, descending=True)[:5]

            retrieved_texts = []
            for max_index in top_3_indices:
                retrieved_text = {}
                retrieved_text["article"] = find_article_name(max_index)
                if 4 <= max_index < len(RAG.doc_texts) - 4:
                    retrieved_text["content"] = '\n'.join(RAG.doc_texts[max_index - 4: max_index + 4])
                elif max_index < 4 and max_index < len(RAG.doc_texts) - 4:
                    retrieved_text["content"] = '\n'.join(RAG.doc_texts[: max_index + 4])
                elif max_index > len(RAG.doc_texts) - 4 and max_index >= 4:
                    retrieved_text["content"] = '\n'.join(RAG.doc_texts[max_index - 4:])
                else:
                    retrieved_text["content"] = '\n'.join(RAG.doc_texts)
                retrieved_texts.append(retrieved_text)
            return retrieved_texts

# 全局记忆管理
class GlobalMemory:
    memories = []

    @staticmethod
    @socketio.on("global_memory_init")
    def memory_init():
        if not os.path.exists(Config.MEMORY_FILE):
            with open(Config.MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)

        with open(Config.MEMORY_FILE, encoding='utf-8') as f:
            memory_data = json.load(f)
            GlobalMemory.memories = [item["memory"] for item in memory_data]

        print('GlobalMemory.memories:', GlobalMemory.memories)

        socketio.emit("global_memory_refresh", [{"index": i, "memory": memory} for i, memory in enumerate(GlobalMemory.memories)])

    @staticmethod
    @socketio.on("add_memory")
    def add_memory(memory):
        """存储单条数据"""
        with open(Config.MEMORY_FILE, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            # 将字符串包装成字典格式
            memory_entry = {
                "memory": memory  # 这里会创建一个包含memory键的字典
            }
            data.append(memory_entry)  # 现在存储的是字典而不是字符串
            f.seek(0)
            json.dump(data, f)
        GlobalMemory.memories.append(memory)
        socketio.emit("global_memory_refresh", [{"index": i, "memory": memory} for i, memory in enumerate(GlobalMemory.memories)])

    @staticmethod
    @socketio.on("remove_memory")
    def remove_memory(index):
        index = int(index)
        GlobalMemory.memories.pop(index)
        with open(Config.MEMORY_FILE, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            data.pop(index)
            # 回写文件
            f.seek(0)
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.truncate()
        socketio.emit("global_memory_refresh", [{"index": i, "memory": memory} for i, memory in enumerate(GlobalMemory.memories)])

    @staticmethod
    def get_memories():
        return '\n'.join(GlobalMemory.memories)

# 多聊天记录管理
class ChatSeries:
    chat_series = []
    current_chat_index = 0
    new_message = True

    @staticmethod
    def init_chat_series():
        ChatSeries.load_local_chat_history()
        ChatSeries.current_chat_index = len(ChatSeries.chat_series) - 1

    @staticmethod
    @socketio.on("chat_list_init")
    def chat_list_init():
        socketio.emit("refresh_chat_list",
                      [{"index": len(ChatSeries.chat_series) - i - 1,
                        "chat_history": chat.chat_history,
                        "stress": False}
                       for i, chat in enumerate(ChatSeries.chat_series[::-1])])

    # 初始加载数据
    @staticmethod
    def load_local_chat_history():
        with open(Config.SERIES_FILE, 'r', encoding='utf-8') as f:
            chat_series = json.load(f)

        for chat_content in chat_series:
            chat = ChatSeries(chat_content)
            ChatSeries.chat_series.append(chat)

    def __init__(self, chat_content=None):
        self.chat_history = []
        if chat_content:
            self.chat_history = chat_content

    @staticmethod
    @socketio.on("choose_chat")
    def choose_chat(index):
        ChatSeries.current_chat_index = int(index)
        socketio.emit("new_chat_window")
        if ChatSeries.current_chat_index == -1:
            socketio.emit("back_to_home")
            return
        else:
            current_chat = ChatSeries.chat_series[ChatSeries.current_chat_index]
            current_chat.refresh_chat_list()
            current_chat.load_history()

    def load_history(self):
        for message in self.chat_history:
            socketio.emit("recognized_text", message["user"])
            socketio.emit("begin_generate_response")
            if "stopReason" in message:
                socketio.emit("retrievedInfo", {
                    "stopReason": message['stopReason'],
                    "stopSequence": message['stopSequence'],
                    "outputTokens": message['outputTokens']
                })
            if "ai" in message:
                socketio.emit("refresh_ai_response", message["ai_html"])
                socketio.emit("ai_response_end", message["ai"])
            else:
                socketio.emit("refresh_ai_response", "输出被打断")
                socketio.emit("ai_response_end", "输出被打断")
        UserInput.update_history()

    @staticmethod
    @socketio.on("delete_chat")
    def delete_chat(index):
        index = int(index)
        if not index == ChatSeries.current_chat_index:
            ChatSeries.chat_series.pop(index)
            with open(Config.SERIES_FILE, 'w', encoding='utf-8') as f:
                json.dump([chat.chat_history for chat in ChatSeries.chat_series], f, ensure_ascii=False, indent=2)
            if ChatSeries.chat_series:
                ChatSeries.current_chat_index = 0
                current_chat = ChatSeries.chat_series[ChatSeries.current_chat_index]
                current_chat.refresh_chat_list()
            else:
                ChatSeries.current_chat_index = -1
        else:
            ChatSeries.chat_series.pop(index)
            if ChatSeries.chat_series:
                ChatSeries.current_chat_index = 0
            else:
                ChatSeries.current_chat_index = -1
            with open(Config.SERIES_FILE, 'w', encoding='utf-8') as f:
                json.dump([chat.chat_history for chat in ChatSeries.chat_series], f, ensure_ascii=False, indent=2)
            socketio.emit("refresh_chat_list",
                          [{"index": len(ChatSeries.chat_series) - i - 1,
                            "chat_history": chat.chat_history,
                            "stress": False}
                           for i, chat in enumerate(ChatSeries.chat_series[::-1])])
            socketio.emit("back_to_home")

    def update_chat_history(self, target_dict):
        if ChatSeries.new_message:
            self.chat_history.append(target_dict)
            ChatSeries.new_message = False
            print("user_input loaded")
        else:
            for key in target_dict:
                self.chat_history[-1][key] = target_dict[key]
                print(f"{key} loaded")
        with open(Config.SERIES_FILE, 'w', encoding='utf-8') as f:
            json.dump([chat.chat_history for chat in ChatSeries.chat_series], f, ensure_ascii=False, indent=2)
        if "ai" in target_dict:
            ChatSeries.new_message = True

    @staticmethod
    @socketio.on("new_chat")
    def create_new_chat():
        new_chat = ChatSeries()
        ChatSeries.chat_series.append(new_chat)
        with open(Config.SERIES_FILE, 'w', encoding='utf-8') as f:
            json.dump([chat.chat_history for chat in ChatSeries.chat_series], f, ensure_ascii=False, indent=2)
        ChatSeries.current_chat_index = len(ChatSeries.chat_series) - 1
        socketio.emit("new_chat_window")
        ChatSeries.chat_series[ChatSeries.current_chat_index].refresh_chat_list()
        UserInput.history = []

    def refresh_chat_list(self):
        socketio.emit("refresh_chat_list",
                      [{"index": len(ChatSeries.chat_series) - i - 1,
                        "chat_history": chat.chat_history,
                        "stress": True if (self == chat) else False}
                       for i, chat in enumerate(ChatSeries.chat_series[::-1])])

# 主函数
if __name__ == "__main__":
    RAG.text_init()
    ChatSeries.init_chat_series()
    GlobalHistory.global_history = GlobalHistory(file_path=r'global_memory/GlobalHistory.json', max_size=10)
    GlobalHistory.load_from_file(r'global_memory/GlobalHistory.json')

    if Config.refresh_rag_documents:
        pdf_process_thread = threading.Thread(target=RAG.process_pdf_for_rag)
        pdf_process_thread.daemon = True  # 设置为守护线程（主线程退出时自动关闭）
        pdf_process_thread.start()
        print('开始处理pdf')

    socketio.run(app, debug=False, allow_unsafe_werkzeug=True)