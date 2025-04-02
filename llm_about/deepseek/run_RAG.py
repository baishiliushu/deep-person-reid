import ollama
import chromadb
import os
from docx import Document
import PyPDF2
0
CHROMA_PATH = os.getenv('CHROMA_PATH', 'chroma')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'local-rag')
TEXT_EMBEDDING_MODEL = os.getenv('TEXT_EMBEDDING_MODEL', 'bge-m3')
DOCUMENT_DIRECTORY = os.getenv('DOCUMENT_DIRECTORY', './knowledge')

files = [f for f in os.listdir(DOCUMENT_DIRECTORY) if os.path.isfile(os.path.join(DOCUMENT_DIRECTORY, f)) and (f.endswith('.docx') or f.endswith('.pdf'))]

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return ' '.join(text)

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
        return text

def extract_text(file_path):
    if file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    else:
        raise ValueError("Unsupported file type")

documents = {}
for file in files:
    file_path = os.path.join(DOCUMENT_DIRECTORY, file)
    try:
        text = extract_text(file_path)
        documents[file] = text
    except Exception as e:
        print(f"Error processing {file}: {e}")

client = chromadb.Client()
collection = client.get_or_create_collection(name=COLLECTION_NAME)
all_ids = collection.get(ids=None)["ids"]
if all_ids:
    collection.delete(ids=all_ids)
for file, d in documents.items():
    response = ollama.embed(model=TEXT_EMBEDDING_MODEL, input=d)
    embeddings = response["embeddings"]
    collection.add(
        ids=[file],
        embeddings=embeddings,
        documents=[d],
        metadatas=[{"file_name": file}]
    )


def main():
    while True:
        user_input = input("请输入问题（输入 'exit' 退出）：")
        if user_input.lower() == 'exit':
            break
        # 生成用户输入的嵌入
        response = ollama.embed(model=TEXT_EMBEDDING_MODEL, input=user_input)
        embeddings = response["embeddings"]
        # 查询最相似的文档
        results = collection.query(
            query_embeddings=embeddings,
            n_results=1
        )
        data = results["documents"][0][0] if results["documents"] else ""
        # 生成回答
        output = ollama.generate(
            model="deepseek-r1:7b",
            prompt=f"""根据以下上下文内容来回答最后的问题，若存在有帮助的回答，则输出知识库中的原文。问题：{user_input} ；有帮助的回答：{data}"""
        )
        print("回答：", output["response"])


if __name__ == '__main__':
    main()
