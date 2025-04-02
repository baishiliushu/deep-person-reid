import requests

def ask_ai(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "deepseek-r1-my:7b",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    return response.json()["response"]


answer = ask_ai("你好")
print(answer)
