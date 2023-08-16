from loguru import logger
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from __init__ import conf

class Embed:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def load_data(self):
        self.df = pd.read_csv('/home/ubuntu/fastapi/catalogo.csv')
        embeddings = self.model.encode(self.df['embed'].tolist())
        self.df['embedding'] = embeddings.tolist()
        self.corpus_embeddings = np.vstack(self.df['embedding'].values)
        self.index = self.build_faiss_index(self.corpus_embeddings)
        
    def build_faiss_index(self, embeddings):
        embeddings = embeddings.astype('float32')
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index
    
    @logger.catch
    def search(self, query: str, top_n: int = 5):
        query_embedding = self.model.encode(query).reshape(1, -1).astype('float32')
        _, top_indices = self.index.search(query_embedding, top_n)
        top_hits = self.df.iloc[top_indices[0]]
        return top_hits[['marca', 'producto']]

if __name__ == '__main__':
    query = "candados"
    busq = Embed()
    busq.search(query=query)