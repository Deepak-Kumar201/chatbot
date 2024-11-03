from django.http import JsonResponse, HttpResponse
from django.http import request
from groq import Groq
import pdfplumber
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings

pdf_text = ""
client = Groq(api_key = "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")
with pdfplumber.open("./chatbot/pdf/9789240100077-eng.pdf") as pdf:
    for page in pdf.pages:
        pdf_text += page.extract_text() + "\n"  

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1365,
    chunk_overlap=256,
    length_function=len
)

embedding = GPT4AllEmbeddings()
chunked_pdf_text = text_splitter.split_text(pdf_text)
print("Text chunked")

embedded_text = FAISS.from_texts(chunked_pdf_text, embedding)
print("Text converted to embeddings")

def save_user_messages(messages, user_id, query, model_response):
    filename = f"user_data/messages_{user_id}.json"
    # messages.append({
    #     "role": "user",
    #     "content": query
    # })
    messages.append({
        "role": "system",
        "content": model_response
    })
    with open(filename, "w") as f:
        json.dump(messages, f, indent=4)

def read_user_messages(user_id):
    try: 
        filename = f"user_data/messages_{user_id}.json"
        data = ""
        with open(filename, 'r') as messages:
            data = json.load(messages)
        if(len(data) >= 10):
            data = data[1:]
        return data
    except: 
        return []
    
def get_similar_doc(query):
    pdf_data = []
    similiar_text = embedded_text.similarity_search(query, k=3)
    for i in similiar_text:
        pdf_data.append({
            "role": "user",
            "content": i.page_content
        })
    return pdf_data

def get_model_body(query, past_messages, pdf_similar_text):
    model_body = list(past_messages)
    for similar_text in pdf_similar_text:
        model_body.append(similar_text)
    model_body.append({
        "role": "user",
        "content": query
    })
    return model_body

def get_query_result(query, user_id):
    past_messages = read_user_messages(user_id)
    pdf_similar_text = get_similar_doc(query)
    model_body = get_model_body(query, past_messages, pdf_similar_text)
    for message in past_messages:
        model_body.append(message)

    chat_completion = client.chat.completions.create(
        messages=model_body,
        model="llama3-8b-8192",
    )
    save_user_messages(past_messages, user_id, query, chat_completion.choices[0].message.content);
    return chat_completion.choices[0].message.content


def getData(request):
    query = request.GET.get('query')
    user_id = request.GET.get("user_id")
    print(f"Request from {user_id} for query {query}")
    model_response = get_query_result(query, user_id)
    return JsonResponse({
        "query": query,
        "answer": model_response
    })
    