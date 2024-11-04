from flask import Flask, request, jsonify
from dotenv import load_dotenv
import chromadb
import os
import uuid
from chromadb.config import Settings
import pandas as pd
import requests

load_dotenv()
app = Flask(__name__)

# chroma_client = chromadb.HttpClient(host='localhost', port=8000, settings=Settings(allow_reset=True))
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
chroma_client.heartbeat()
# chroma_client.reset()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb.utils.embedding_functions as embedding_functions
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas import EvaluationDataset
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness,
)
from ragas import evaluate
import ast

from filestorage import getPresignedUrls
from minio.commonconfig import Tags
from filestorage import setMetaTags

# volname:/chroma/chroma => for volume mounting while running the container
# -e ALLOW_RESET=true => for allowing reset of the database

def docLoad(url):
    loader = PyPDFLoader(url)
    docs = loader.load()
    print(len(docs))
    return docs

def textSplitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(len(splits))    
    return splits

def embedAndStore(splits):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="text-embedding-3-small"
                )
    collection = chroma_client.get_or_create_collection(name="aggr", embedding_function=openai_ef)
    collection.add(ids=[str(uuid.uuid4()) for split in splits], documents=[split.page_content for split in splits])
    return "done successfully"

def retreive(query):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=os.getenv("OPENAI_API_KEY"),
                        model_name="text-embedding-3-small"
                    )
    collection = chroma_client.get_collection(name="aggr", embedding_function=openai_ef)
    results = collection.query(
        query_texts=[query], # Chroma will embed this for you
        n_results=2 # how many results to return,
    )

    return results

PROMPT_TEMPLATE = """
Answer the question based on the following context, and refuse to answer if the question is out of scope, answer like -> I can't answer that question as it seems out of scope.
{context}
 - -
Answer the question based on the above context, dont start the answer with something like -> Acc to the ctx, ... <- , use the context: {question}
"""

def queryLLM(ctx, query_text):
    context_text = "\n\n - -\n\n".join([doc for doc in ctx["documents"][0]])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    model = ChatOpenAI()
    res = model.predict(prompt)
    
    return res

def gemerateEvalTestCases(doc):
    loader = PyPDFLoader(doc)
    docs = loader.load()
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    dataset = generator.generate_with_langchain_docs(docs, testset_size=10)
    df = dataset.to_pandas()
    df.to_csv('test.csv')

def answerQuery(query):
    qctxts = retreive(query=query)
    res = queryLLM(qctxts, query_text=query)
    return res

def answerTestCases(testcasefile):
    df = pd.DataFrame({
        "user_input": [],
        "reference_contexts": [],
        "reference": [],
        "retreived_contexts": [],
        "response": []
    })

    tcf = pd.read_csv(testcasefile)
    for index, row in tcf.iterrows():
        user_input = row['user_input']
        reference_contexts = row['reference_contexts']
        reference = row['reference']
        retreived_contexts = retreive(query=row['user_input'])
        response = queryLLM(retreived_contexts, row['user_input'])
        
        df = pd.concat([df, pd.DataFrame({
            "user_input": user_input,
            "reference_contexts": reference_contexts,
            "reference": reference,
            "retrieved_contexts": retreived_contexts["documents"],
            "response": response
        })] , ignore_index=True)
            
    df.to_csv('test_results.csv')

def evaluateTestCases(testcasefile):
    tcf = pd.read_csv(testcasefile)
    tcf = tcf.drop(tcf.columns[0], axis=1)
    # drop the column with the incorrect spelling of retrieved_contexts
    tcf = tcf.drop(columns=['retreived_contexts'])
    
    idict = tcf.to_dict()

    refctx = idict['reference_contexts']
    refArr = []
    
    for key in refctx:
        refArr.append(ast.literal_eval(refctx[key]))
    
    idict['reference_contexts'] = refArr
    
    retctx = idict['retrieved_contexts']
    retArr = []
    
    for key in retctx:
        retArr.append(ast.literal_eval(retctx[key]))
    
    idict['retrieved_contexts'] = retArr
    idict['response'] = [idict['response'][key] for key in idict['response']]
    idict['user_input'] = [idict['user_input'][key] for key in idict['user_input']]
    idict['reference'] = [idict['reference'][key] for key in idict['reference']]
    # print(idict["response"])
    
    for x in idict["reference_contexts"][3]:
        print(x)
    
    # eval_data = Dataset.from_dict(idict)
    
    # metrics = [
    #     faithfulness,
    #     answer_relevancy,
    #     context_precision,
    #     context_recall,
    #     answer_similarity,
    #     answer_correctness,
    # ]
    # results = evaluate(eval_data, metrics)    
    # df = results.to_pandas()
    # df.to_csv('eval_results.csv')
    
# docs = docLoad('./hci.pdf')
# splits = textSplitter(docs)
# res = embedAndStore(splits=splits)

# qctxts = retreive("How do i secure the atm?")
# print(qctxts)

# res = queryLLM(qctxts, "How do i secure the atm?")
# print(res)

# gemerateEvalTestCases('./hci.pdf')

# evaluateTestCases('test_results.csv')

@app.route('/')
def init():
    return 'Init'

@app.route('/upload', methods=['POST'])
def upload_file():
    
    file = request.files['file']
    if file:

        url = getPresignedUrls(file.filename)
        print(url)
        response = requests.put(url, data=file)
        # tags = Tags.new_object_tags()
        # tags["uid"] = "123"
        # tags["pid"] = "456"
        
        # setMetaTags(file.filename, tags)
        
        if response.status_code == 200:
            print(response)
            return jsonify({'done': 'Uploaded'}), 200
        else:
            return jsonify({'error': 'Upload failed'}), 500
    return jsonify({'error': 'No file uploaded'}), 400

from filestorage import getAllFiles
from filestorage import getPresignedGetUrls

# @app.route('/files', methods=['GET'])
# def get_files():
#     files = getAllFiles()
#     file_urls = [getPresignedGetUrls(file.object_name) for file in files]
#     return jsonify(file_urls), 200

if __name__ == '__main__':
    app.run(debug=True)
