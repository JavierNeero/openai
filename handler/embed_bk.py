from loguru import logger
import openai
import guidance
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from __init__ import conf

class Embed:
    def __init__(self):
        self.df = pd.read_csv('/home/ubuntu/fastapi/embebido.csv', converters={'embedding': eval})
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.df['embedding'] = self.df['embedding'].apply(lambda x: torch.Tensor(x))
        self.corpus_embeddings = torch.stack(self.df['embedding'].tolist())
        self.corpus_embeddings = util.normalize_embeddings(self.corpus_embeddings)
    
    @logger.catch
    def search(self, query: str, top_n: int = 5, similarity_threshold=0.4):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_n)
        top_hits = [hit for hit in hits[0] if hit['score'] >= similarity_threshold]
        top_indices = [hit['corpus_id'] for hit in top_hits]

        if len(top_indices) == 0:
            return None
        
        return self.df.loc[top_indices, ['marca', 'producto']]

if __name__ == '__main__':
    query = "candados"
    busq = Embed()
    busq.search(query=query)