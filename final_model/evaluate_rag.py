"""
RAG evaluation with automatic metrics (Ass5 style)
"""
import json
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics import f1_score, accuracy_score
import time
import re
from dotenv import load_dotenv
import os

print("Starting evaluation script...")
import sys
sys.stdout.flush()

# Load environment
load_dotenv("rag.env")
hf_token = os.getenv("HF_TOKEN")

print("Loading evaluation data...")
sys.stdout.flush()

data_path = "qa_data.json"

# Load evaluation data
with open(data_path, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

print("\n=== SANITY CHECK 1 ===\n")
print(f"Example question: {eval_data[0]['question']}")
print(f"Gold label: {eval_data[0]['gold_label']}")
print(f"\nExample course context (first 500 chars): {eval_data[0]['context'][:500]}...")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

print("\n=== SANITY CHECK 2 ===\n")
text = "This is a test document."
query_result = embeddings.embed_query(text)
print(f"Embedding for test document:", query_result[:3])

# Initialize LLM
# Load LLM
model_name = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    dtype="auto", 
    use_auth_token=hf_token
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    do_sample=False,
    return_full_text=False
)



llm = HuggingFacePipeline(pipeline=pipe)

print("\nLLM LOADED\n.")


# Split documents into chunks
# Can be extended to test different chunk sizes
chunk_sizes = [200]
resulting_chunk_accuracy = []

for cs in chunk_sizes:

    print(f"  Chunk size option: {cs}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cs,
        chunk_overlap=100
    )

    docs = []
    seen_courses = set()
    for qa in eval_data:
        if qa['course_code'] and qa['course_code'] not in seen_courses:
            full_doc = Document(
                page_content=qa['context'],
                metadata={"course_code": qa['course_code']}
            )
            chunks = text_splitter.split_documents([full_doc])
            
            # Prepend course code to each chunk for better retrieval
            for chunk in chunks:
                chunk.page_content = f"{qa['course_code']}\n{chunk.page_content}"
            
            docs.extend(chunks)
            seen_courses.add(qa['course_code'])

    print("\n=== SANITY CHECK 3 ===\n")
    print(f"\nTotal chunks created: {len(docs)}")
    print("\nSample chunks:")
    for i in range(min(3, len(docs))):
        print(f"\nChunk {i+1}:")
        print(f"  Content: {docs[i].page_content[:300]}...")
        print(f"  Metadata: {docs[i].metadata}")

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="course_eval"
    )
    print("\nVector store created")


    # Test retrieval
    print("\n=== SANITY CHECK 4 ===\n")
    test_q = "Is the course DAT246 compulsory?"
    results = vectorstore.similarity_search_with_score(test_q, k=3)
    for res, score in results:
        print(f"\n[SIM={score:.3f}] {res.page_content[:200]}... [{res.metadata}]")

    # Smart retrieval with course code filtering
    def extract_course_code(text):
        match = re.search(r"\b[A-Z]{3}\d{3}\b", text)
        return match.group(0) if match else None


    def retrieve_with_filter(question):
        #course_code = extract_course_code(question)
        #if course_code:
        #    return vectorstore.similarity_search(
        #        question, 
        #        k=5,
        #        filter={"course_code": course_code}
        #    )
        return vectorstore.similarity_search(question, k=5)

    filtered_retriever = RunnableLambda(lambda q: retrieve_with_filter(q))


    # Build RAG chain
    def combine_context(docs):
        return "\n\n".join([d.page_content for d in docs])

    template = """You are answering questions about Chalmers University courses. Use ONLY the information in the context below - do not use prior knowledge.

    Context:
    {context}

    Question: {question}

    Answer with only "Yes" or "No":"""

    prompt = ChatPromptTemplate.from_template(template)

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

    # Test RAG chain
    test_question = eval_data[0]['question']
    print("\n=== SANITY CHECK 5 ===\n")
    print(f"Question: {test_question}")

    result = rag_chain.invoke(test_question)

    print(f"\nContext (first 500 chars):")
    print(result["context"][:500] + "...")

    print(f"\nAnswer:")
    print(result["answer"])


    # EVALUATION

    print("\n\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    # Extract yes/no from model output
    def extract_yes_no(answer):
        answer_lower = answer.lower().strip()
        if re.match(r'^yes\b', answer_lower):
            return 'yes'
        elif re.match(r'^no\b', answer_lower):
            return 'no'
        first_sentence = answer_lower.split('.')[0]
        if 'yes' in first_sentence and 'no' not in first_sentence:
            return 'yes'
        elif 'no' in first_sentence and 'yes' not in first_sentence:
            return 'no'
        return None

    # TASK 1: Evaluating RAG with context
    print("\n" + "-"*80)
    print("TASK 1: Evaluating RAG with context")
    print("-"*80)

    rag_predictions = []
    rag_gold_labels = []

    print(f"Processing {len(eval_data)} questions...")
    start_time = time.time()

    for i, qa in enumerate(eval_data):
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {i+1}/{len(eval_data)} ({elapsed:.1f}s elapsed)")
        
        try:
            result = rag_chain.invoke(qa['question'])
            predicted_label = extract_yes_no(result["answer"])
            
            if predicted_label is not None:
                rag_predictions.append(predicted_label)
                rag_gold_labels.append(qa['gold_label'])
        except Exception as e:
            print(f"  Error on question {i+1}: {e}")

    elapsed_time = time.time() - start_time

    print(f"\nRAG Results (completed in {elapsed_time:.1f}s):")
    print(f"  Total questions: {len(eval_data)}")
    print(f"  Valid answers: {len(rag_predictions)} ({100*len(rag_predictions)/len(eval_data):.1f}%)")
    if len(rag_predictions) > 0:
        rag_accuracy = accuracy_score(rag_gold_labels, rag_predictions)
        rag_f1 = f1_score(rag_gold_labels, rag_predictions, pos_label='yes')
        print(f"  Accuracy: {rag_accuracy:.4f}")
        print(f"  F1-score: {rag_f1:.4f}")

    # TASK 2: Baseline (no context)
    print("\n" + "-"*80)
    print("TASK 2: Evaluating baseline (no context)")
    print("-"*80)

    baseline_template = """Question: {question}

    Answer with only "Yes" or "No":"""
    baseline_prompt = ChatPromptTemplate.from_template(baseline_template)
    baseline_chain = baseline_prompt | llm | StrOutputParser()

    baseline_predictions = []
    baseline_gold_labels = []

    print(f"Processing {len(eval_data)} questions...")
    start_time = time.time()

    for i, qa in enumerate(eval_data):
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {i+1}/{len(eval_data)} ({elapsed:.1f}s elapsed)")
        
        try:
            answer = baseline_chain.invoke({"question": qa['question']})
            predicted_label = extract_yes_no(answer)
            
            if predicted_label is not None:
                baseline_predictions.append(predicted_label)
                baseline_gold_labels.append(qa['gold_label'])
        except Exception as e:
            print(f"  Error on question {i+1}: {e}")

    elapsed_time = time.time() - start_time

    print(f"\nBaseline Results (completed in {elapsed_time:.1f}s):")
    print(f"  Total questions: {len(eval_data)}")
    print(f"  Valid answers: {len(baseline_predictions)} ({100*len(baseline_predictions)/len(eval_data):.1f}%)")
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



    # TASK 3: Retrieval quality
    print("\n" + "-"*80)
    print("TASK 3: Evaluating retrieval quality")
    print("-"*80)

    retrieval_hits = 0
    retrieval_total = 0

    for i, qa in enumerate(eval_data):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(eval_data)}")
        
        gold_course = qa['course_code']
        if not gold_course:
            continue
        
        try:
            retrieved_docs = filtered_retriever.invoke(qa['question'])
            retrieved_courses = [doc.metadata.get('course_code') for doc in retrieved_docs]
            
            if gold_course in retrieved_courses:
                retrieval_hits += 1
            retrieval_total += 1
        except Exception as e:
            continue


    retrieval_accuracy = retrieval_hits / retrieval_total if retrieval_total > 0 else 0
    print(f"\nRetrieval Results:")
    print(f"  Total questions: {retrieval_total}")
    print(f"  Gold document retrieved: {retrieval_hits} ({100*retrieval_accuracy:.1f}%)")


    print("\n" + "-"*80)
    print("Task 4: Examining errors:")
    print("-"*80)
    
    print(f"\nErrors:")
    errors = []
    for qa in eval_data:
        result = rag_chain.invoke(qa['question'])
        predicted = extract_yes_no(result["answer"])
        if predicted != qa['gold_label']:
            errors.append({
                'question': qa['question'],
                'predicted': predicted,
                'gold': qa['gold_label'],
                'context': result['context'][:200]
            })
    print(f"  Total questions that were wrong: {len(errors)}")
    print(f" Details:")
    for error in errors:
        print(f"  Question: {error['question']}")
        print(f"  Predicted: {error['predicted']}, Gold: {error['gold']}")
        print(f"  Context (first 200 chars): {error['context']}")


    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

    resulting_chunk_accuracy.append((cs, rag_accuracy))


print("\nChunk Size vs RAG Accuracy:")
for cs, acc in resulting_chunk_accuracy:
    print(f"  Chunk Size: {cs} => RAG Accuracy: {acc:.4f}")



