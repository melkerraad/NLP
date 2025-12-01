#make sure to activate dat450_venv before running this script, will not work otherwise

import pandas as pd
tmp_data = pd.read_json("ori_pqal.json").T
# some labels have been defined as "maybe", only keep the yes/no answers
tmp_data = tmp_data[tmp_data.final_decision.isin(["yes", "no"])]

documents = pd.DataFrame({"abstract": tmp_data.apply(lambda row: (" ").join(row.CONTEXTS+[row.LONG_ANSWER]), axis=1),
             "year": tmp_data.YEAR})
questions = pd.DataFrame({"question": tmp_data.QUESTION,
             "year": tmp_data.YEAR,
             "gold_label": tmp_data.final_decision,
             "gold_context": tmp_data.LONG_ANSWER,
             "gold_document_id": documents.index})

print(f"\n\n=== SANITY CHECK 1===")
print(f"Question: {questions.iloc[0].question}")
print(f"Document abstract: {documents.iloc[0].abstract}")

#Initialize the model, load llama from huggingface
import os
from langchain_community.llms import HuggingFacePipeline

hf_token = os.getenv('HF_TOKEN')    #Load token from bash
model=HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.2-3B", 
    task="text-generation",
    device=0, #Use GPU
    pipeline_kwargs={"return_full_text": False, "max_new_tokens": 150, "do_sample": False},
    model_kwargs={"use_auth_token": hf_token}
)

#Testing the model on a basic prompt
print(f"\n\n=== SANITY CHECK 2===")
print("Question: What is the average area of a parking lot?\n")
print(f"Answer: {model.invoke("What is the average area of a parking lot?")}")
print(model.invoke("What is the average area of a parking lot?"))

#Initializing th embedding model 

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
text = "This is a test document."
query_result = embeddings.embed_query(text)
print("\n\n=== SANITY CHECK 3===")
print("Embedding for test document:")
print(query_result[:3])
doc_result = embeddings.embed_documents([text])




#Split the text into chunks smartly and save the metadatas
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

#Creating document objects that can be used by LangChain, holds page_content and metadata
metadatas = [{"id": idx} for idx in documents.index]
texts = text_splitter.create_documents(texts=documents.abstract.tolist(), metadatas=metadatas)

print(f"\nTotal chunks created: {len(texts)}")
print("\nSample chunks:")
for i in range(min(10, len(texts))):
    print(f"\nChunk {i+1}:")
    print(f"  Content: {texts[i].page_content[:500]}...")
    print(f"  Metadata: {texts[i].metadata}")


from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="texts",
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

print("Vector store created")
vector_store.add_documents(texts)

#See if it retrieves stuff from relevant abstracts:

testsim= "How can cancer be diagnosed early and treated?"
results_simsearch = vector_store.similarity_search(
    testsim,
    k=4,
)

for res in results_simsearch:
    print(f"* {res.page_content} [{res.metadata}]")
    print("---")

print("should have printed the results by now")

# Step 4: Build RAG pipeline (Option B)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

#Define retriever (fetch 3 documents for better context)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) #formatting function to combine docs into context string

#Define template and create prompt
template = """Answer the question based only on the context below. Be very brief and specific (1-3 sentences).
Context:
{context}
Question:
{question}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

#Define RunnableParallel object with context and question
runnable_parallel_object = RunnableParallel(
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
)

#Construct the retrieval chain (text generation only). The chain receives the dict from RunnableParallel, so prompt goes first.
chain = (
    prompt
    | model
    | StrOutputParser()
)

#Combine using assign method. This passes the {context, question} dict through prompt, then to model.
rag_chain = runnable_parallel_object.assign(answer=chain)

# Sanity check
test_question = questions.iloc[0].question
print(f"\n\n=== SANITY CHECK ===")
print(f"Question: {test_question}")

result = rag_chain.invoke(test_question)

print(f"\nContext:")
print(result["context"])

print(f"\nAnswer:")
print(result["answer"])


# ===== STEP 5: EVALUATION =====
print("\n\n" + "="*80)
print("STEP 5: RAG EVALUATION")
print("="*80)

from sklearn.metrics import f1_score, accuracy_score
import re
import time

def extract_yes_no(answer):
    """Extract yes/no from model answer. Returns 'yes', 'no', or None if unclear."""
    answer_lower = answer.lower().strip()
    
    # Look for explicit yes/no at start
    if re.match(r'^yes\b', answer_lower):
        return 'yes'
    elif re.match(r'^no\b', answer_lower):
        return 'no'
    
    # Look for yes/no anywhere in first sentence
    first_sentence = answer_lower.split('.')[0]
    if 'yes' in first_sentence and 'no' not in first_sentence:
        return 'yes'
    elif 'no' in first_sentence and 'yes' not in first_sentence:
        return 'no'
    
    return None  # Ambiguous or neither

# SAMPLE SIZE - set to small number for testing, None for full dataset
SAMPLE_SIZE = None  # Change to None to process all questions

eval_questions = questions.head(SAMPLE_SIZE) if SAMPLE_SIZE else questions

# Task 1: Evaluate RAG with context
print("\n" + "-"*80)
print("TASK 1: Evaluating RAG with context")
print("-"*80)

rag_predictions = []
rag_gold_labels = []
rag_valid_indices = []

print(f"Processing {len(eval_questions)} questions...")
start_time = time.time()

for i, (idx, row) in enumerate(eval_questions.iterrows()):
    if i % 10 == 0:
        elapsed = time.time() - start_time
        print(f"  Progress: {i}/{len(eval_questions)} ({elapsed:.1f}s elapsed)")
    
    question = row.question
    gold_label = row.gold_label
    
    try:
        result = rag_chain.invoke(question)
        answer = result["answer"]
        predicted_label = extract_yes_no(answer)
        
        if predicted_label is not None:  # Only count valid answers
            rag_predictions.append(predicted_label)
            rag_gold_labels.append(gold_label)
            rag_valid_indices.append(idx)
    except Exception as e:
        print(f"  Error on question {idx}: {e}")
        continue

elapsed_time = time.time() - start_time
print(f"\nRAG Results (completed in {elapsed_time:.1f}s):")
print(f"  Total questions: {len(eval_questions)}")
print(f"  Valid answers: {len(rag_predictions)} ({100*len(rag_predictions)/len(eval_questions):.1f}%)")
if len(rag_predictions) > 0:
    rag_accuracy = accuracy_score(rag_gold_labels, rag_predictions)
    rag_f1 = f1_score(rag_gold_labels, rag_predictions, pos_label='yes')
    print(f"  Accuracy: {rag_accuracy:.4f}")
    print(f"  F1-score: {rag_f1:.4f}")


# Task 2: Baseline without context
print("\n" + "-"*80)
print("TASK 2: Evaluating baseline (no context)")
print("-"*80)

baseline_template = """Answer the following question with only "Yes" or "No":
Question: {question}
Answer:"""
baseline_prompt = ChatPromptTemplate.from_template(baseline_template)
baseline_chain = baseline_prompt | model | StrOutputParser()

baseline_predictions = []
baseline_gold_labels = []
baseline_valid_indices = []

print(f"Processing {len(eval_questions)} questions...")
start_time = time.time()

for i, (idx, row) in enumerate(eval_questions.iterrows()):
    if i % 10 == 0:
        elapsed = time.time() - start_time
        print(f"  Progress: {i}/{len(eval_questions)} ({elapsed:.1f}s elapsed)")
    
    question = row.question
    gold_label = row.gold_label
    
    try:
        answer = baseline_chain.invoke({"question": question})
        predicted_label = extract_yes_no(answer)
        
        if predicted_label is not None:
            baseline_predictions.append(predicted_label)
            baseline_gold_labels.append(gold_label)
            baseline_valid_indices.append(idx)
    except Exception as e:
        print(f"  Error on question {idx}: {e}")
        continue

elapsed_time = time.time() - start_time
print(f"\nBaseline Results (completed in {elapsed_time:.1f}s):")
print(f"  Total questions: {len(eval_questions)}")
print(f"  Valid answers: {len(baseline_predictions)} ({100*len(baseline_predictions)/len(eval_questions):.1f}%)")
if len(baseline_predictions) > 0:
    baseline_accuracy = accuracy_score(baseline_gold_labels, baseline_predictions)
    baseline_f1 = f1_score(baseline_gold_labels, baseline_predictions, pos_label='yes')
    print(f"  Accuracy: {baseline_accuracy:.4f}")
    print(f"  F1-score: {baseline_f1:.4f}")

    print(f"\nComparison:")
    print(f"  RAG Accuracy: {rag_accuracy:.4f} vs Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"  RAG F1: {rag_f1:.4f} vs Baseline F1: {baseline_f1:.4f}")
    print(f"  Did retrieval help? {'YES' if rag_accuracy > baseline_accuracy else 'NO'} (accuracy)")
    print(f"  Did retrieval help? {'YES' if rag_f1 > baseline_f1 else 'NO'} (F1)")


# Task 3: Evaluate retrieval quality (gold document matching)
print("\n" + "-"*80)
print("TASK 3: Evaluating retrieval quality")
print("-"*80)

retrieval_hits = 0
retrieval_total = 0

print(f"Checking if gold documents are retrieved for {len(eval_questions)} questions...")
start_time = time.time()

for i, (idx, row) in enumerate(eval_questions.iterrows()):
    if i % 10 == 0:
        elapsed = time.time() - start_time
        print(f"  Progress: {i}/{len(eval_questions)} ({elapsed:.1f}s elapsed)")
    
    question = row.question
    gold_doc_id = row.gold_document_id
    
    try:
        # Get retrieved documents
        retrieved_docs = retriever.invoke(question)
        retrieved_ids = [doc.metadata['id'] for doc in retrieved_docs]
        
        if gold_doc_id in retrieved_ids:
            retrieval_hits += 1
        
        retrieval_total += 1
    except Exception as e:
        print(f"  Error on question {idx}: {e}")
        continue

retrieval_accuracy = retrieval_hits / retrieval_total if retrieval_total > 0 else 0
print(f"\nRetrieval Results:")
print(f"  Total questions: {retrieval_total}")
print(f"  Gold document retrieved: {retrieval_hits} ({100*retrieval_accuracy:.1f}%)")


# Task 4: Inspect some examples
print("\n" + "-"*80)
print("TASK 4: Inspecting sample results")
print("-"*80)

import random
random.seed(42)
sample_indices = random.sample(range(len(eval_questions)), min(5, len(eval_questions)))

for i, idx in enumerate(sample_indices):
    row = eval_questions.iloc[idx]
    print(f"\n{'='*60}")
    print(f"EXAMPLE {i+1}/{len(sample_indices)}")
    print(f"{'='*60}")
    print(f"Question: {row.question}")
    print(f"Gold label: {row.gold_label}")
    print(f"Gold document ID: {row.gold_document_id}")
    
    try:
        # Get retrieval results
        retrieved_docs = retriever.invoke(row.question)
        retrieved_ids = [doc.metadata['id'] for doc in retrieved_docs]
        
        print(f"\nRetrieved document IDs: {retrieved_ids}")
        print(f"Gold document retrieved: {'YES' if row.gold_document_id in retrieved_ids else 'NO'}")
        
        # Get RAG answer
        result = rag_chain.invoke(row.question)
        answer = result["answer"]
        predicted_label = extract_yes_no(answer)
        
        print(f"\nRetrieved context (first 300 chars):")
        print(result["context"][:300] + "...")
        
        print(f"\nModel answer: {answer}")
        print(f"Predicted label: {predicted_label}")
        print(f"Correct: {'YES' if predicted_label == row.gold_label else 'NO'}")
        
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)



