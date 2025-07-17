import os
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# from langchain_community.document_loaders import PyPDFLoader

# file_path = "/content/Tahmid_Rahman_Cv-2.pdf"
# loader = PyPDFLoader(file_path)
# docs = loader.load()
# docs[0]

# --- 3. Setup Embeddings ---
embedding_model = HuggingFaceEmbeddings(model_name="l3cube-pune/bengali-sentence-similarity-sbert")

# --- 4. Prepare Chunks with Metadata ---
agriculture_chunks = [
    ("ধানের চাষের জন্য ভালো বীজ নির্বাচন করা জরুরি।", {"category": "agriculture"}),
    ("ফসলের জন্য সঠিক সেচ পদ্ধতি ব্যবহার করা উচিত।", {"category": "agriculture"}),
    ("জৈব সার ব্যবহারে মাটির গুণমান ভালো থাকে।", {"category": "agriculture"}),
    ("গম চাষের জন্য ঠাণ্ডা আবহাওয়া উপযোগী।", {"category": "agriculture"}),
    ("পোকামাকড় নিয়ন্ত্রণে বায়োলজিক্যাল পদ্ধতি ব্যবহার করুন।", {"category": "agriculture"}),
    ("ফসল ঘরের পরে জমি বিশ্রাম দেওয়া গুরুত্বপূর্ণ।", {"category": "agriculture"}),
    ("আধুনিক প্রযুক্তি ব্যবহারে কৃষির উৎপাদন বাড়ে।", {"category": "agriculture"}),
    ("আবহাওয়ার পূর্বাভাস দেখে চাষাবাদ করা উচিত।", {"category": "agriculture"}),
    ("শাকসবজি চাষে কম পানি প্রয়োজন।", {"category": "agriculture"}),
    ("কৃষিতে ক্ষুদ্রঋণ সহায়তা প্রদান করে সরকার।", {"category": "agriculture"})
]

fruit_chunks = [
    ("আম পাকা হলে এটি হলুদ রঙ ধারণ করে।", {"category": "fruits"}),
    ("কলা একটি পুষ্টিকর ফল।", {"category": "fruits"}),
    ("লিচু গরম আবহাওয়ায় ভালো হয়।", {"category": "fruits"}),
    ("তরমুজ শরীর ঠান্ডা রাখতে সাহায্য করে।", {"category": "fruits"}),
    ("কমলা ভিটামিন সি এর ভালো উৎস।", {"category": "fruits"}),
    ("আনারস হজমে সাহায্য করে।", {"category": "fruits"}),
    ("জাম অনেক অ্যান্টিঅক্সিডেন্ট সমৃদ্ধ।", {"category": "fruits"}),
    ("পেঁপে রক্তচাপ নিয়ন্ত্রণে সাহায্য করে।", {"category": "fruits"}),
    ("ড্রাগন ফল বর্তমানে খুব জনপ্রিয়।", {"category": "fruits"}),
    ("আমরসা রুচি বৃদ্ধি করে।", {"category": "fruits"})
]

bank_chunks = [
    ("ক্রেডিট কার্ড খোলার জন্য জাতীয় পরিচয়পত্র প্রয়োজন।", {"category": "bank"}),
    ("ব্যাংক অ্যাকাউন্ট খোলার জন্য ফরম পূরণ করতে হয়।", {"category": "bank"}),
    ("চেকবই পাওয়ার জন্য অ্যাকাউন্ট অ্যাকটিভ থাকতে হবে।", {"category": "bank"}),
    ("এটিএম কার্ড ব্যবহার করে টাকা তোলা যায়।", {"category": "bank"}),
    ("মোবাইল ব্যাংকিং সার্ভিস বর্তমানে খুব জনপ্রিয়।", {"category": "bank"}),
    ("সঞ্চয়ী অ্যাকাউন্টে সুদের হার কম।", {"category": "bank"}),
    ("ব্যাংক লোন নেওয়ার জন্য জামিনদার লাগতে পারে।", {"category": "bank"}),
    ("ইন্টারনেট ব্যাংকিংয়ের মাধ্যমে ঘরে বসেই লেনদেন করা যায়।", {"category": "bank"}),
    ("ক্রেডিট কার্ডে সীমা নির্ধারণ করা থাকে।", {"category": "bank"}),
    ("অনলাইন ব্যাংকিং সেবার জন্য আলাদা পাসওয়ার্ড লাগে।", {"category": "bank"})
]

all_chunks = agriculture_chunks + fruit_chunks + bank_chunks

documents = [Document(page_content=text, metadata=meta) for text, meta in all_chunks]

# --- 5. Setup Vector Store ---
vector_store = FAISS.from_documents(documents, embedding_model)

# --- 6. Define Metadata Filter Function ---
def filter_by_metadata(query, category):
    """
    Filter vector store by category metadata first, then perform similarity search.
    """
    print("\n[Metadata Filtering] Category:", category)
    filtered_docs = [doc for doc in documents if doc.metadata['category'] == category]
    for doc in filtered_docs:
        print(" -", doc.page_content)

    if not filtered_docs:
        return []
    temp_vector_store = FAISS.from_documents(filtered_docs, embedding_model)
    similar_docs = temp_vector_store.similarity_search(query, k=2)

    print("\n[Similarity Search Results for Query]", query)
    for doc in similar_docs:
        print(" >>", doc.page_content)

    return similar_docs


# --- 7. Setup GitHub-hosted OpenAI LLM (via OpenAI SDK) ---

token = os.environ['GITHUB_TOKEN']
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

# --- 8. Define Final RAG Chain ---
def ask_faq_bot(user_question: str, category: str):
    print("\n[User Question]", user_question)
    docs = filter_by_metadata(user_question, category)
    context = "\n".join([doc.page_content for doc in docs])
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "তুমি একজন সহায়ক বাংলা সহকারী। শুধুমাত্র নিচের প্রাসঙ্গিক তথ্য থেকে উত্তর দাও। যদি প্রশ্নের উত্তর এতে না থাকে, বলো 'দুঃখিত, এই বিষয়ে আমার জানা নেই।" + context,
            },
            {
                "role": "user",
                "content": user_question,
            }
        ],
        temperature=0.7,
        top_p=1.0,
        model=model
    )
    return response.choices[0].message.content


# --- 9. Test Questions (Each in Different Cell Below) ---

# Cell 1
question = "আমি কীভাবে একটি ক্রেডিট কার্ড খুলতে পারি?"
category = "bank"
response = ask_faq_bot(question, category)
print("উত্তর:", response)

#---------Find catergory from user input and call the function---------

# --- 10. Add Category Router

def detect_category_llm(question):
    """
    Use LLM to determine appropriate category
    """
    system_msg = "তুমি একটি শ্রেণিবিন্যাসকারী এজেন্ট। নিচের প্রশ্নটি পড়ে বলো এটি কোন ক্যাটাগরির মধ্যে পড়ে: agriculture, fruits, bank। শুধুমাত্র ক্যাটাগরির নাম এক শব্দে ইংরেজিতে উত্তর দাও।"
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question}
        ],
        model=model,
        temperature=0,
        top_p=1.0
    )
    category = response.choices[0].message.content.strip().lower()
    print("[LLM Router] Category:", category)
    return category

question = "আমি কীভাবে একটি ক্রেডিট কার্ড খুলতে পারি?"
category = detect_category_llm(question)
response = ask_faq_bot(question, category)
print("উত্তর:", response)