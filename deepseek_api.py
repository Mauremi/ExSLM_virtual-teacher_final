import requests
import json

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer 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",
        "Content-Type": "application/json",
        "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
        "X-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
    },
    data=json.dumps({
        "model": "qwen/qwen3-1.7b:free",
        "messages": [
            {
                "role": "user",
                "content": "What is the meaning of life?"
            }
        ],

    })
)

# 检查响应状态码
if response.status_code == 200:
    # 解析响应内容
    response_data = response.json()
    print("Response:", response_data)
else:
    print(f"Error: {response.status_code}")
    print("Response:", response.text)