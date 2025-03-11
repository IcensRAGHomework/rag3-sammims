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
    chroma_client = chromadb.PersistentClient(path = dbpath)

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

            document = row["HostWords"]
            # update data to ChromaDB
            # https://python.langchain.com/docs/integrations/vectorstores/chroma/
            # https://docs.trychroma.com/docs/collections/add-data
            collection.add(
                ids = str(idx),
                metadatas = row_values,
                documents = document
            )

    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    collection = generate_hw01()

    # combined filters
    # https://docs.trychroma.com/docs/querying-collections/metadata-filtering
    filters = []
    if city:
        filters.append(
            {
                'city': {
                    '$in': city
                }
            }
        )
    if store_type:
        filters.append(
            {
                'type': {
                    '$in': store_type
                }
            }
        )
    if start_date:
        filters.append(
            {
                'date': {
                    '$gte': int(start_date.timestamp())
                }
            }
        )
    if end_date:
        filters.append(
            {
                'date': {
                    '$lte': int(end_date.timestamp())
                }
            }
        )

    # 1) get 10 results based on query and filters
    match_query_results = collection.query(
        query_texts = [question],
        n_results = 10,
        where = {
            '$and': filters
        }
    )
    print("match_query_results-----")
    print(match_query_results)
    # 2) keep results with similarity >= 0.8 from the 10 results matching query
    good_similarity_results = []
    for i in range(len(match_query_results['ids'])):
        # zip() function pairs elements from both lists together
        for distance, metadata in zip(match_query_results['distances'][i], match_query_results['metadatas'][i]):
            cosine_similarity = 1 - distance
            if cosine_similarity >= 0.8:
                good_similarity_results.append(metadata["name"])
    print("good_similarity_results-----")
    print(good_similarity_results)

    return good_similarity_results
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    collection = generate_hw01()

    # add new_store_name column for the row data matching store_name
    results = collection.query(
        query_texts=[store_name],
        n_results=1
    )
    for i in range(len(results['ids'])):
        for metadata in results['metadatas'][i]:
            if metadata['name'] == store_name:
                metadata['new_store_name'] = new_store_name
                collection.update(
                    ids = results['ids'][i],
                    metadatas = [metadata]
                )

    # combined filters
    # https://docs.trychroma.com/docs/querying-collections/metadata-filtering
    filters = []
    if city:
        filters.append(
            {
                'city': {
                    '$in': city
                }
            }
        )
    if store_type:
        filters.append(
            {
                'type': {
                    '$in': store_type
                }
            }
        )

    # 1) get 10 results based on query and filters
    match_query_results = collection.query(
        query_texts = [question],
        n_results = 10,
        where = {
            '$and': filters
        }
    )
    print("match_query_results-----")
    print(match_query_results)
    # 2) keep results with similarity >= 0.8 from the 10 results matching query
    #    - if new_store_name column exist, use it as name column in results
    #    - sort with similarity in descending order
    good_similarity_results = []
    for i in range(len(match_query_results['ids'])):
        # zip() function pairs elements from both lists together
        for distance, metadata in zip(match_query_results['distances'][i], match_query_results['metadatas'][i]):
            cosine_similarity = 1 - distance
            if cosine_similarity >= 0.8:
                # use new_store_name instead if any
                if "new_store_name" in metadata:
                    good_similarity_results.append((metadata["new_store_name"], cosine_similarity))
                else:
                    good_similarity_results.append((metadata["name"], cosine_similarity))
    print(good_similarity_results)
    # sort the final results with similarity(x[1], 1 means the second element in the result row) in descending order
    good_similarity_results.sort(key = lambda x: x[1], reverse = True)
    print("good_similarity_results-----")
    print(good_similarity_results)

    sorted_name_results_list = [name for name, _ in good_similarity_results]
    print("sorted_name_results_list-----")
    print(sorted_name_results_list)
    return sorted_name_results_list;
    
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
    print("hw03_1----------------------------------------------------------------")
    generate_hw01()
    print("\n")

    print("hw03_2----------------------------------------------------------------")
    generate_hw02("我想要找有關茶餐點的店家", ["宜蘭縣", "新北市"], ["美食"], datetime.datetime(2024, 4, 1), datetime.datetime(2024, 5, 1))
    print("\n")

    print("hw03_3----------------------------------------------------------------")
    generate_hw03("我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵", "耄饕客棧", "田媽媽（耄饕客棧）", ["南投縣"], ["美食"])
    print("\n")