import datetime
import chromadb
import traceback
import pandas as pd

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

collection_name = "TRAVEL"
collection_metadata = {"hnsw:space": "cosine"}

csv_file_name = "COA_OpenData.csv"

def generate_hw01():
    # edge sqlite database
    chroma_client = chromadb.PersistentClient(path=dbpath)

    # create embedding function
    # https://docs.trychroma.com/integrations/embedding-models/openai
    # model_name: str = "text-embedding-ada-002" by default
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name'],
        model_name = gpt_emb_config['model']
    )

    # create collection (database create table)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function = openai_ef,
        metadata=collection_metadata
    )

    # create if not exist
    if collection.count() == 0:
        # reads a CSV (Comma-Separated Values) file and loads it into a Pandas DataFrame (df).
        df = pd.read_csv(csv_file_name)
        print("columns: " + str(df.columns)) # df.columns is Index object
    
        for idx, row in df.iterrows():
            # access row values by column name
            row_values = {
                "file_name": csv_file_name,
                "name": row["Name"],
                "type": row["Type"],
                "address": row["Address"],
                "tel": row["Tel"],
                "city": row["City"],
                "town": row["Town"],
                "date": int(datetime.datetime.strptime(row['CreateDate'], '%Y-%m-%d').timestamp())  # unix timeStamp
            }
            print(str(idx) + str(row_values))
            print("\n")

            # update data to ChromaDB
            # https://python.langchain.com/docs/integrations/vectorstores/chroma/
            collection.add(
                ids=[str(idx)],
                metadatas=[row_values],
                documents=[row["HostWords"]]
            )
        return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    pass
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection

if __name__ == "__main__":
    print("hw03_1")
    print("----------------------------------------------------------------")
    generate_hw01()
    print("\n")
