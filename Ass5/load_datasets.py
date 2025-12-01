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

print(questions.iloc[0].question)
print(documents.iloc[0].abstract)

#Initialize the model, load llama from huggingface
import os
from langchain_community.llms import HuggingFacePipeline

hf_token = os.getenv('HF_TOKEN')    #Load token from bash
model=HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.2-3B", 
    task="text-generation",
    device=0, #Use GPU
    pipeline_kwargs={"return_full_text": False},
    model_kwargs={"use_auth_token": hf_token}
)

#Testing the model on a basic prompt
print(model.invoke("What is the average area of a parking lot?"))

#Initializing th embedding model 

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
text = "This is a test document."
query_result = embeddings.embed_query(text)
print(query_result[:3])
doc_result = embeddings.embed_documents([text])




#Split the text into chunks smartly and save the metadatas
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
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


