from flask import Flask, request, jsonify
from dotenv import load_dotenv
import chromadb
import os
import uuid
import pandas as pd
import requests
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
import uuid
from sqlalchemy import inspect

load_dotenv(override=True)
app = Flask(__name__)

print(os.getenv("RANDOM"))

app.config['SECRET_KEY'] = 'your-secret-key'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5432/pdfspace'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
 
# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    spaces = db.relationship('Space', backref='owner', lazy=True)
    uploaded_pdfs = db.relationship('PDF', backref='uploader', lazy=True)

class Space(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    pdfs = db.relationship('PDF', secondary='space_pdf', backref='spaces')

class PDF(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    object_id = db.Column(db.String(100), unique=True, nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Association tables
space_pdf = db.Table('space_pdf',
    db.Column('space_id', db.Integer, db.ForeignKey('space.id'), primary_key=True),
    db.Column('pdf_id', db.Integer, db.ForeignKey('pdf.id'), primary_key=True)
)

space_user = db.Table('space_user',
    db.Column('space_id', db.Integer, db.ForeignKey('space.id'), primary_key=True),
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True)
)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


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
Answer the question based on the following context
{context}
 - -
Answer the question based on the above context, dont start the answer with something like -> Acc to the ctx, ... <- , use the context: {question}
"""

def queryLLM(ctx, query_text):
    context_text = "\n\n - -\n\n".join([doc for doc in ctx["documents"][0]])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    print(prompt)
    
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
    # print(qctxts)
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

# res = answerQuery("How do the security features and usability enhancements in ATM systems compare across different reports, considering aspects like user-centered design, biometrics for security, and interface design improvements?")
# print(res)

# gemerateEvalTestCases('./hci.pdf')

# evaluateTestCases('test_results.csv')

@app.route('/')
def init():
    return 'Init'

@app.route('/getuserfiles', methods=['GET'])
@login_required
def get_user_files():
    user_files = PDF.query.filter_by(uploaded_by=current_user.id).all()
    file_urls = [getPresignedGetUrls(file.object_id) for file in user_files]
    return jsonify(file_urls), 200

@app.route('/upload', methods=['POST'])
@login_required

def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    space_id = request.form.get('space_id')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Your existing file upload logic here
        object_id = str(uuid.uuid4())  # Generate unique object ID
        
        url = getPresignedUrls(object_id)
        print(url)
        response = requests.put(url, data=file)

        print(response.status_code) 

        # if response.status_code != 200:
            # return jsonify({'error': 'Failed to upload file'}), 500

        # Create PDF record
        pdf = PDF(
            object_id=object_id,
            filename=file.filename,
            uploaded_by=current_user.id
        )
        db.session.add(pdf)

        # If space_id is provided, add PDF to space
        if space_id:
            space = Space.query.get(space_id)
            if space and (space.created_by == current_user.id or current_user.id in [u.id for u in space.users]):
                space.pdfs.append(pdf)

        db.session.commit()
        return jsonify({
            'message': 'File uploaded successfully',
            'object_id': object_id
        }), 200

    return jsonify({'error': 'File type not allowed'}), 400

from filestorage import getAllFiles
from filestorage import getPresignedGetUrls

# @app.route('/files', methods=['GET'])
# def get_files():
#     files = getAllFiles()
#     file_urls = [getPresignedGetUrls(file.object_name) for file in files]
#     return jsonify(file_urls), 200

# Authentication routes
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 400
    
    user = User(
        email=data['email'],
        password_hash=generate_password_hash(data['password'])
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(email=data['email']).first()
    if user and check_password_hash(user.password_hash, data['password']):
        login_user(user)
        return jsonify({'message': 'Logged in successfully'})
    return jsonify({'error': 'Invalid credentials'}), 401

# Space management endpoints
@app.route('/spaces', methods=['POST'])
@login_required
def create_space():
    data = request.json
    space = Space(
        name=data['name'],
        created_by=current_user.id
    )
    db.session.add(space)
    db.session.commit()
    return jsonify({'message': 'Space created successfully', 'space_id': space.id}), 201

@app.route('/spaces/<int:space_id>/share', methods=['POST'])
@login_required
def share_space(space_id):
    space = Space.query.get_or_404(space_id)
    if space.created_by != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json
    user = User.query.filter_by(email=data['email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    statement = space_user.insert().values(space_id=space_id, user_id=user.id)
    db.session.execute(statement)
    db.session.commit()
    return jsonify({'message': 'Space shared successfully'}), 200

# Add this function to check allowed file types
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}  # Add your allowed extensions here
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create a directory for temporary files if it doesn't exist
TEMP_DIR = 'tempfiles'
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/index_upload', methods=['POST'])
@login_required
def index_and_upload():
    user_files = PDF.query.filter_by(uploaded_by=current_user.id).all()
    for file in user_files:
        presigned_url = getPresignedGetUrls(file.object_id)
        
        # Download the file
        response = requests.get(presigned_url)
        if response.status_code == 200:
            temp_file_path = os.path.join(TEMP_DIR, file.filename)
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(response.content)
                
                docs = docLoad(temp_file_path) 
                splits = textSplitter(docs)
                print(splits)
                embedAndStore(splits) 
        else:
            return jsonify({'error': f'Failed to download file {file.filename}'}), 500

    return jsonify({'message': 'All user files indexed and uploaded successfully'}), 200

if __name__ == '__main__':
    with app.app_context():
        print("Checking for existing database tables...")
        inspector = inspect(db.engine)
        
        # Check if the User table exists
        if not inspector.has_table('user'):
            print("Creating database tables...")
            db.create_all()  # Create tables if they do not exist
            print("Database tables created.")
        else:
            print("Database tables already exist.")
    app.run(debug=True)