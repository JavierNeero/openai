from loguru import logger
import openai
import guidance
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import re
from __init__ import conf

gpt3 = guidance.llms.OpenAI("gpt-3.5-turbo")

class Embed:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def load_data(self):
        self.df = pd.read_csv('/home/ubuntu/fastapi/catalogo700.csv')
        self.corpus_embeddings = self.model.encode(self.df['combinado'].tolist())
        faiss.normalize_L2(self.corpus_embeddings)
        self.index = faiss.IndexFlatIP(self.corpus_embeddings.shape[1])
        self.index.add(self.corpus_embeddings)
    
    @logger.catch
    def search(self, query: str, top_n: int = 5):
        query_embedding = self.model.encode(query).reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        _, indices = self.index.search(query_embedding, top_n)
        top_hits = self.df.iloc[indices[0]][['marca', 'producto']]
        return top_hits

class QueryData:
    def __init__(self):
        self.msg = None

    @logger.catch
    def chat_data(self, question: str, semantic_search: callable):
        bot = guidance("""
        {{#system~}}
        Eres Tulia, una asesora especialista en diseño y arquitectura de interiores, dedicada a brindar un excepcional y personalizado servicio al cliente que trabaja para Tul (www.tul.io).
        {{~/system}}

        {{#user~}}
        Clasifica en 3 palabras la intención en la frase del usuario, asi: si quiere hablar con un asesor responde transferir con asesor, si quiere comprar materiales responde compra de materiales, si quiere consultar por producto responde consulta por producto, si quiere consultar por marca responde consulta por marca, si quiere consultar producto y marca responde consulta producto marca, si no puedes clasificar la intención responder intencion sin clasificar, responde con máximo 3 palabras, sin puntos y todo en minuscula
        Frase: {{question}}
        {{~/user}}

        {{#assistant~}}
        {{gen 'intencion' temperature=0 max_tokens=10}}
        {{~/assistant}}

        {{#user~}}
        Extrae el producto y la marca de la frase, sino encuentras producto no respondas y sino encuentras marca no respondas
        Frase: {{question}}
        {{~/user}}

        {{#assistant~}}
        {{gen 'producto' temperature=0.0 max_tokens=10}}
        {{~/assistant}}

        {{#if (== intencion "consulta por producto")}}
        {{#user~}}
        Genera respuesta a la Consulta teniendo la información del producto en el Catalogo
        Consulta: {{question}}
        Catalogo: {{semantic_search producto}}
        {{~/user}}
        {{else}}
        {{#user~}}
        Responde en máximo 30 palabras
        {{question}}
        {{~/user}}
        {{/if}}

        {{#assistant~}}
        {{gen 'answer' temperature=0.7 max_tokens=500}}
        {{~/assistant}}

        """, llm=gpt3)

        return bot(question=question, semantic_search=semantic_search)

if __name__ == '__main__':
    print('Testing...')
