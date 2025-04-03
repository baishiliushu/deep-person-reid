import requests
import ollama


def ask_ai(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "deepseek-r1-my:7b",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    return response.json()["response"]


def ask_ai_generate(prompt):
    # 直接调用ollama中的generate函数
    response = ollama.generate(
        model="deepseek-r1-my:7b",
        prompt=prompt,
        stream=False
    )
    return response['response']


# answer = ask_ai("你好")
answer = ask_ai_generate("你好")
print(answer)
