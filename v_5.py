# -*- coding: utf-8 -*-
"""
State-of-the-Art GraphRAG (HyDE + Communities) - Google Colab
=============================================================
Novel Mechanisms:
1. HyDE (Hypothetical Document Embeddings): hallucinate-then-retrieve.
2. Hierarchical Community Detection: Global context awareness.
3. PageRank Centrality: Identifying authoritative nodes.
4. Semantic Attention: Edge-level relevance filtering.

INSTRUCTIONS:
1. Copy into Colab cell.
2. Runtime > Change runtime type > T4 GPU.
3. Run.
"""

import os
import subprocess
import sys
import networkx as nx
import matplotlib.pyplot as plt
import torch
import textwrap
import numpy as np

# --- 1. Environment ---
def install_dependencies():
    print("Installing dependencies (2-3 mins)...")
    pkgs = [
        "transformers", "accelerate", "bitsandbytes", "langchain",
        "langchain-community", "networkx", "sentence-transformers", 
        "faiss-gpu", "scipy"
    ]
    for p in pkgs:
        try:
            __import__(p.replace("-", "_").split("==")[0])
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", p, "-q"])
    print("Dependencies ready.")

try:
    import google.colab
    IN_COLAB = True
    install_dependencies()
except ImportError:
    IN_COLAB = False

# --- 2. Model Loading ---
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer, util
import faiss
from networkx.algorithms import community

print("\nLoading AI Models...")

# Embedding Model (for Vector Space)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# LLM (Zephyr-7b) 4-bit Quantized
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "HuggingFaceH4/zephyr-7b-beta"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
except Exception as e:
    print(f"Model load error: {e}")
    sys.exit(1)

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer,
    max_new_tokens=512, temperature=0.1, top_p=0.95, repetition_penalty=1.15
)
llm = HuggingFacePipeline(pipeline=pipe)

# --- 3. SOTA Graph Engine ---

class SOTAGraphRAG:
    def __init__(self, llm_engine, embedder_model):
        self.llm = llm_engine
        self.embedder = embedder_model
        self.graph = nx.Graph() # Undirected for community detection
        self.vector_index = None
        self.nodes_list = []
        self.pagerank = {}
        self.community_summaries = {} # {id: "summary text"}

    def extract_triplets(self, text):
        """Extracts Entity-Relation-Entity using LLM."""
        tmpl = """<|system|>
        Extract knowledge triplets (Subject | Relation | Object).
        - Use precise entities.
        - Output ONE triplet per line.
        </s>
        <|user|>
        {text}
        </s>
        <|assistant|>"""
        chain = LLMChain(llm=self.llm, prompt=PromptTemplate(template=tmpl, input_variables=["text"]))
        resp = chain.run(text=text)
        
        triplets = []
        for line in resp.split('\n'):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 3: triplets.append(parts)
        return triplets

    def detect_communities(self):
        """
        Novelty: Uses Greedy Modularity to find 'Topics' in the graph.
        Generates a summary for the top 3 largest communities (Global Context).
        """
        if self.graph.number_of_nodes() < 5: return
        
        print("  > Detecting Graph Communities (Global Context)...")
        # Community Detection
        communities = community.greedy_modularity_communities(self.graph)
        
        # Summarize top 3 clusters
        for i, comm in enumerate(communities[:3]):
            subgraph_nodes = list(comm)
            # Create a mini-text representation of the cluster
            cluster_text = ", ".join(subgraph_nodes[:20]) # Limit to 20 nodes to save tokens
            
            tmpl = """<|system|>
            Summarize the common theme of these entities in 2 sentences.
            </s>
            <|user|>
            Entities: {entities}
            </s>
            <|assistant|>"""
            chain = LLMChain(llm=self.llm, prompt=PromptTemplate(template=tmpl, input_variables=["entities"]))
            summary = chain.run(entities=cluster_text).strip()
            self.community_summaries[i] = summary
            print(f"    Cluster {i}: {summary}")

    def build_system(self, text_corpus):
        print(f"Processing {len(text_corpus)} chars...")
        chunks = textwrap.wrap(text_corpus, 1000)
        
        # 1. Build Graph
        for chunk in chunks:
            for s, r, o in self.extract_triplets(chunk):
                self.graph.add_edge(s, o, relation=r)
                
        # 2. Centrality & Communities
        if self.graph.number_of_nodes() > 0:
            try: self.pagerank = nx.pagerank(self.graph)
            except: self.pagerank = {n:1.0 for n in self.graph.nodes()}
            
            self.detect_communities()
            
            # 3. Vector Index
            self.nodes_list = list(self.graph.nodes())
            emb = self.embedder.encode(self.nodes_list)
            faiss.normalize_L2(emb)
            self.vector_index = faiss.IndexFlatIP(emb.shape[1])
            self.vector_index.add(emb)
            print(f"Indexed {len(self.nodes_list)} nodes.")
        else:
            print("Graph empty.")

    def generate_hyde(self, query):
        """
        Novelty: HyDE (Hypothetical Document Embeddings).
        Generates a fake answer to search for semantic matches.
        """
        tmpl = """<|system|>
        Write a short, hypothetical passage that answers this question perfectly.
        Do not use real facts, just hallucinate a plausible structure.
        </s>
        <|user|>
        Question: {query}
        </s>
        <|assistant|>"""
        chain = LLMChain(llm=self.llm, prompt=PromptTemplate(template=tmpl, input_variables=["query"]))
        return chain.run(query=query)

    def retrieve(self, query):
        # 1. Generate HyDE Vector
        fake_ans = self.generate_hyde(query)
        print(f"\n[HyDE Hallucination]: {fake_ans[:100]}...")
        q_emb = self.embedder.encode(fake_ans) # Embed the ANSWER, not the query
        
        # 2. Vector Search (Anchors)
        faiss.normalize_L2(q_emb.reshape(1, -1))
        D, I = self.vector_index.search(q_emb.reshape(1, -1), 3)
        
        candidates = set()
        # 3. Graph Traversal + Attention
        for idx in I[0]:
            if idx < len(self.nodes_list):
                node = self.nodes_list[idx]
                edges = self.graph.edges(node, data=True)
                for u, v, d in edges:
                    # Score edge relevance (Attention)
                    edge_str = f"{u} {d.get('relation','related')} {v}"
                    edge_emb = self.embedder.encode(edge_str)
                    attn = util.cos_sim(q_emb, edge_emb).item()
                    
                    # Boost by centrality
                    boost = self.pagerank.get(u,0) + self.pagerank.get(v,0)
                    score = attn + boost
                    
                    candidates.add((score, edge_str))
        
        # Sort by score
        sorted_facts = sorted(list(candidates), key=lambda x: x[0], reverse=True)
        local_context = [x[1] for x in sorted_facts[:7]]
        
        # 4. Global Context (Communities)
        global_context = list(self.community_summaries.values())
        
        return local_context, global_context

    def query(self, user_q):
        local, glob = self.retrieve(user_q)
        
        ctx_str = "Global Themes:\n" + "\n".join(glob) + "\n\nSpecific Facts:\n" + "\n".join(local)
        
        tmpl = """<|system|>
        Answer using the Context.
        - 'Global Themes' are high-level summaries.
        - 'Specific Facts' are direct graph connections.
        Context:
        {ctx}
        </s>
        <|user|>
        {q}
        </s>
        <|assistant|>"""
        chain = LLMChain(llm=self.llm, prompt=PromptTemplate(template=tmpl, input_variables=["ctx", "q"]))
        return chain.run(ctx=ctx_str, q=user_q)

# --- 4. Main ---
if __name__ == "__main__":
    rag = SOTAGraphRAG(llm, embedder)
    
    # Text Input
    text = ""
    if IN_COLAB:
        from google.colab import files
        print("Upload text file:")
        up = files.upload()
        if up: text = list(up.values())[0].decode('utf-8', 'ignore')
    else:
        text = input("Enter text/path: ")
        if os.path.exists(text): text = open(text).read()

    if text:
        rag.build_system(text)
        while True:
            q = input("\nQuery: ")
            if q.lower() == 'exit': break
            print(f"Answer: {rag.query(q)}")
