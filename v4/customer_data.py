import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings

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