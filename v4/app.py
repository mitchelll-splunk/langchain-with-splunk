import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from flask import Flask, request
from langchain.globals import set_debug

set_debug(True)

app = Flask(__name__)
model = ChatOpenAI(model="gpt-3.5-turbo")

file_path = (
   "./customers-1000.csv"
)

loader = CSVLoader(file_path=file_path)
customer_data = loader.load()

embeddings_model = OpenAIEmbeddings()

weaviate_client = weaviate.connect_to_local()

db = WeaviateVectorStore.from_documents(
   customer_data,
   embedding=embeddings_model,
   client=weaviate_client,
)

results = db.similarity_search(
   "Which customers are associated with the company Cherry and Sons?"
)

for result in results:
   print("\n")
   print(result.page_content)

store = {}
config = {"configurable": {"session_id": "test"}}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)

@app.route("/askquestion", methods=['POST'])
def ask_question():

    data = request.json
    question = data.get('question')
    print(question)

    # find the documents most similar to the question that we can pass as context
    context = db.similarity_search(question)

    response = with_message_history.invoke(
        [
            SystemMessage(
                content=f'Use the following pieces of context to answer the question: {context}'
            ),
            HumanMessage(
                content=question
            )
        ],
        config=config
    )

    return response.content

