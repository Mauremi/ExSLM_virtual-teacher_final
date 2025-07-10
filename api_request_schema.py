api_request_list = {
    'alibaba.qwen/qwen3-1.7b:free':{
        "modelId": "alibaba.qwen/qwen3-1.7b:free",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "model": "qwen/qwen3-1.7b:free",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the meaning of life?"
                }
            ],
        }
    },
    'anthropic.claude-3-sonnet-20240229-v1:0': {
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "messages": "",
            "system": "",
            "max_tokens": 10000,
            "temperature": 0.1,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": [
                "\n\nHuman:"
            ],
            "anthropic_version": "bedrock-2023-05-31"
        }
    },
    'anthropic.claude-3-5-sonnet-20240620-v1:0': {
        "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "messages": "",
            "system": "",
            "max_tokens": 10000,
            "temperature": 0.1,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": [
                "\n\nHuman:"
            ],
            "anthropic_version": "bedrock-2023-05-31"
        }
    },
    'meta.llama3-70b-instruct-v1': {
        "modelId": "meta.llama3-70b-instruct-v1:0",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "prompt": "",
            "max_gen_len": 512,
            "temperature": 0.1,
            "top_p": 0.9
        }
    }
}

def get_model_ids():
    return list(api_request_list.keys())
