import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.documents import Document
import time
import re
from dotenv import load_dotenv
import os


load_dotenv("rag.env")

hf_token = os.getenv("HF_TOKEN")

with open("courses_clean.json", "r", encoding="utf-8") as f:
    courses = json.load(f)

docs = []

for course in courses:
    full_text = (
        f"{course['course_code']}\n"  # prefix improves embedding search
        f"Course name: {course['course_name']}\n"
        f"Type: {course['course_type']}\n"
        f"Description: {course['description']}\n"
        f"URL: {course['url']}"
    )
    docs.append(Document(
        page_content=full_text,
        metadata={"course_code": course["course_code"]}
    ))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

all_splits = text_splitter.split_documents(docs)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
)
t2 = time.time()
document_ids = vectorstore.add_documents(documents=all_splits)
t3 = time.time()

def extract_course_code(text):
    match = re.search(r"\b[A-Z]{3}\d{3}\b", text)
    return match.group(0) if match else None


def retrieve_with_filter(question):
    course_code = extract_course_code(question)

    if course_code:
        #print(f"> Detected course code: {course_code} — applying metadata filter")
        return vectorstore.similarity_search(
            question, 
            k=5,
            filter={"course_code": course_code}
        )

    print("> No course code detected — using standard similarity search")
    return vectorstore.similarity_search(question, k=5)

filtered_retriever = RunnableLambda(lambda q: retrieve_with_filter(q))

model_name = "meta-llama/Llama-3.2-1B-Instruct"
#model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype="auto", use_auth_token=hf_token)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    do_sample=False,
    return_full_text=True
)

llm = HuggingFacePipeline(pipeline=pipe)

def combine_context(docs):
    return "\n\n".join([d.page_content for d in docs])

template = """You are a helpful AI assistant. 
Answer the question **only using the context below**. Be very brief (1-3 sentences).
Do not answer any other questions mentioned in the context. Ignore examples not related to the user's question.

Context:
{context}

Question:
{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Build the RAG structure
runnable_parallel_object = RunnableParallel(
    {
        "context": filtered_retriever | combine_context,
        "question": RunnablePassthrough(),
    }
)

chain = (
    prompt
    | llm
    | StrOutputParser()
)

rag_chain = runnable_parallel_object.assign(answer=chain)

questions = [
    "Who is the Examiner for the course DAT695?",
    "What are the possible grades for the course DAT695?",
    "What are the course specific prerequisites for the course TMA947?",
    "What content will the course TMA947 cover?",
    "What block schedule is DAT246 in, answer with one letter?",
    "Is the course DAT246 compulsory?",
    "What is the full course name for DAT441?",
    "Does the course DAT441 have any programming prerequisites?",
    "What are the course specific prerequisites for DAT441?",
    "What is the aim of the course DAT465?",
    "Is there a written exam in the course DAT465?",
    "Who is the examiner for the course DAT625?",
    "Is english 6 required for the course DAT625?",
    "Can i get the grade 5 in the course EEN100?",
    "What are the course specific requirements for EEN100?",
    "Do I need to know Linear Algebra to enrole the course FFR105?",
    "What litterature is used in the course FFR105?",
    "Has the examiner written the litterature for the course FFR105?",
    "Is the course FFR135 compulsory?",
    "What is the content of the course FFR135?",
    "Is MVE188 a math specific course?",
    "What is the content of the course MVE188?",
    "Is the course RRY025 open for exchange students?",
    "What is the name and mail of the examiner for the course RRY025?",
    "Will I learn about vanishing gradients and convolutional neural networks in the course SSY340?",
    "What litterature is used in SSY340?",
    "Is there a written exam in the course TIN093?",
    "What litterature is used in the course TIN093?",
    "Who is the examiner for the course TMA265 and what is him/hers phone number?",
    "Is the examiner of the course TMA882 a woman?",
    "What are the course specific prerequisites for TMA882?",
    "Will i learn about Bayesian inference in the course MVE550?",
    "What type of exam is it in the course MVE550?",
    "Is DAT450 a pure math course?",
    "Can i enrole DAT450 without any programming skills?",
    "What are the course specific prerequisites for DAT450?",
    "Is the aim of the course DAT450 mainly how to use CNNs?",
    "Summarize the content of the course DAT450.",
    "Is there a written exam in the course DAT450?",
    "How will the examination work in the course DAT450?",
    "Who is the examiner for the course DAT570?",
    "Is there a written exam in the course DAT570?",
    "What is the litterature used in DAT570?",
    "Do I need to know linear algebra to enrole in the course EEN020?",
    "What litterature is used in the course EEN020?",
    "Is there a written exam in the course EEN020?",
    "How am I graded in the course EEN020?",
    "Is the examiner for MVE095 a female?",
    "What are the course specific prerequisites for MVE095?",
    "Is there any assignments in the course MVE095?,"
]


question = "Is the examiner for MVE095 a female?"

t0 = time.time()
result = rag_chain.invoke(question)
t1 = time.time()
print("\n--- RAG Answer ---")
print(result["answer"])

print("\n--- Timing ---")
print(f"Time to invoke = {t1-t0}")
print(f"Time to load vectorstore= {t3-t2}")

"""
for ind, question in enumerate(questions, start=1):
    result = rag_chain.invoke(question)
    print(f"Q{ind}: {question}")
    print(f"A{ind}: {result['answer']}")
    print("")
"""
