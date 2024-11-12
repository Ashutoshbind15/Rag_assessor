from flask import Flask, request, jsonify
from dotenv import load_dotenv
import chromadb
import os
import uuid
import pandas as pd
import requests
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
import uuid
from sqlalchemy import inspect
from libs.auth import login_init, login_manager
from flask_cors import CORS, cross_origin
from functools import wraps

load_dotenv(override=True)
app = Flask(__name__)

print(os.getenv("RANDOM"))

app.config['SECRET_KEY'] = 'your-secret-key'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5432/pdfspace'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_init(app)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# todo: add migrations

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    spaces_created = db.relationship('Space', backref='owner', lazy=True)
    uploaded_pdfs = db.relationship('PDF', backref='uploader', lazy=True)
    iterations = db.relationship('Iteration', backref='creator', lazy=True)

class Space(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    iterations = db.relationship('Iteration', secondary='space_iteration', backref='spaces', lazy=True)
    users = db.relationship('User', secondary='space_user', backref='spaces_shared')
    
class PDF(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    object_id = db.Column(db.String(100), unique=True, nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
class Iteration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    embedding_function = db.Column(db.String(255), nullable=False)
    distance_function = db.Column(db.String(255), nullable=False)
    vector_store = db.Column(db.String(255), nullable=False)
    assessments = db.relationship('Assessment', backref='iteration', lazy=True)
    files = db.relationship('PDF', secondary='iteration_pdf', backref='iterations')
    collection_name = db.Column(db.String(255), nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Assessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    iteration_id = db.Column(db.Integer, db.ForeignKey('iteration.id'), nullable=False)
    testcasefile = db.Column(db.String(255), nullable=False)
    resultsfile = db.Column(db.String(255), nullable=False)
    # todo: add metrics

# Association tables

space_user = db.Table('space_user',
    db.Column('space_id', db.Integer, db.ForeignKey('space.id'), primary_key=True),
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True)
)

iteration_pdf = db.Table('iteration_pdf',
    db.Column('iteration_id', db.Integer, db.ForeignKey('iteration.id'), primary_key=True),
    db.Column('pdf_id', db.Integer, db.ForeignKey('pdf.id'), primary_key=True)
)

space_iteration = db.Table('space_iteration',
    db.Column('space_id', db.Integer, db.ForeignKey('space.id'), primary_key=True),
    db.Column('iteration_id', db.Integer, db.ForeignKey('iteration.id'), primary_key=True)
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

@app.route('/test', methods=['GET'])
def test():
    # test json response
    return jsonify({'message': 'Test successful'}), 200

def session_reqd(f):
    @wraps(f)
    @cross_origin(supports_credentials=True)
    @login_required
    def decorated_function(*args, **kwargs):
        return f(*args, **kwargs)
    return decorated_function

@app.route('/getuserfiles', methods=['GET'])
@session_reqd
def get_user_files():
    user_files = PDF.query.filter_by(uploaded_by=current_user.id).all()
    
    res = [{
            'id': file.id,
            'filename': file.filename
        } for file in user_files]
    return jsonify(res), 200

@app.route('/upload', methods=['POST'])
@session_reqd

def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    iteration_id = request.form.get('iteration_id')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
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
        # todo: allow upload for multiple file types
        if iteration_id:
            iteration = Iteration.query.get(iteration_id)
            if iteration:
                iteration.files.append(pdf)

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
@cross_origin(supports_credentials=True)
def login():
    data = request.json
    user = User.query.filter_by(email=data['email']).first()
    if user and check_password_hash(user.password_hash, data['password']):
        login_user(user)
        return jsonify({'message': 'Logged in successfully'})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/logout', methods=['POST'])
@session_reqd
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/profile', methods=['GET'])
@session_reqd
def profile():
    user = User.query.get_or_404(current_user.id)
    return jsonify({
        'email': user.email,
        'id': user.id 
    }), 200
    
# todo: facilitate client side file upload from the urls
# flow: 
# 1. get presigned url for the file 
# 2. upload the file to the url
# 3. send the object id to the server
# 4. some server side validation and then add to db, or call a webhook after adding to the filestore, if there's a util for that

# iteration management eps

@app.route('/iterations', methods = ["POST"])
@session_reqd
def create_iteration():
    data = request.json
    
    iteration = Iteration(
        embedding_function=data['embedding_function'],
        distance_function=data['distance_function'] if 'distance_function' in data else 'l2 squared norm',
        vector_store=data['vector_store'] if 'vector_store' in data else 'chroma',
        created_by=current_user.id
    )
    
    collection = None
    
    if data['embedding_function'] == 'openai':
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        collection = chroma_client.create_collection(name=data['collection_name'], embedding_function=openai_ef)
    
    elif data['embedding_function'] == 'sentence-transformers':
        ef = embedding_functions.DefaultEmbeddingFunction()
        collection = chroma_client.create_collection(name=data['collection_name'], embedding_function=ef)
    
    else:
        return jsonify({'error': 'Invalid embedding function'}), 400 
    
    iteration.collection_name = collection.name
    db.session.add(iteration)
    db.session.commit()
    return jsonify({'message': 'Iteration created successfully', 'iteration_id': iteration.id}), 201

@app.route('/iterations/<int:iteration_id>', methods=['GET'])
@session_reqd
def get_iteration(iteration_id):
    iteration = Iteration.query.get_or_404(iteration_id)
    return jsonify({'iteration': iteration}), 200

@app.route('/iterations', methods=['GET'])
@session_reqd
def get_iterations():
    iterations = Iteration.query.filter_by(created_by=current_user.id).all()
    return jsonify({'iterations': iterations}), 200

# Space management endpoints
@app.route('/spaces', methods=['POST'])
@session_reqd
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
@session_reqd
def share_space(space_id):
    space = Space.query.get_or_404(space_id)
    if space.created_by != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json
    emails = data['emails']
    for email in emails:
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        space.users.append(user)
    
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
@session_reqd
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

# Custom error handler for unauthorized access
@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({'error': 'Unauthorized access'}), 401

def drop_all_tables():
    print("Dropping all database tables...")
    db.drop_all()
    print("All tables dropped successfully")

if __name__ == '__main__':
    with app.app_context():
        # drop_all_tables() 
        
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