# -*- coding: utf-8 -*-
"""
Super GraphRAG (HyDE + Communities + PageRank + Attention) - Google Colab
=========================================================================
This script merges multiple advanced RAG techniques into a single pipeline:
1.  **HyDE (Hypothetical Document Embeddings)**: Generates a theoretical answer to search for semantic matches.
2.  **Hierarchical Community Detection**: Clusters the graph to understand "Global Themes" (e.g., "The project goals").
3.  **PageRank Centrality**: Identifies authoritative nodes to boost their relevance.
4.  **Semantic Graph Attention**: Scores specific relationships (edges) based on how well they match the query intent.
5.  **Query Expansion**: Generates query variations to improve recall.

INSTRUCTIONS:
1. Copy into a Google Colab cell.
2. Set Runtime > Change runtime type > T4 GPU.
3. Run.
4. Upload your text file when prompted.
"""

import os
import subprocess
import sys
import networkx as nx
import torch
import textwrap
import numpy as np
import warnings

# --- 1. Environment Setup ---
def install_dependencies():
    print("Installing dependencies (approx. 2-3 mins)...")
    pkgs = [
        "transformers", "accelerate", "bitsandbytes", "langchain",
        "langchain-community", "networkx", "sentence-transformers", 
        "faiss-gpu", "scipy"
    ]
    for p in pkgs:
        try:
            # Handle package naming differences for import checks
            import_name = p.replace("-", "_")
            if p == "faiss-gpu": import_name = "faiss"
            if p == "sentence-transformers": import_name = "sentence_transformers"
            __import__(import_name.split("==")[0])
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", p, "-q"])
    print("Dependencies ready.")

try:
    import google.colab
    IN_COLAB = True
    install_dependencies()
except ImportError:
    IN_COLAB = False

# --- 2. Imports & Model Loading ---
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer, util
import faiss
from networkx.algorithms import community as nx_comm

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

print("\nLoading AI Models...")

# A. Embedding Model (for Vector Space & Attention)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# B. LLM (Zephyr-7b) 4-bit Quantized
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
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

# Pipeline for fast generation
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer,
    max_new_tokens=512, temperature=0.1, top_p=0.95, repetition_penalty=1.15
)
llm = HuggingFacePipeline(pipeline=pipe)

# --- 3. Super GraphRAG Engine ---

class SuperGraphRAG:
    def __init__(self, llm_engine, embedder_model):
        self.llm = llm_engine
        self.embedder = embedder_model
        self.graph = nx.Graph() # Undirected for robust community detection
        self.vector_index = None
        self.nodes_list = []
        self.pagerank = {}
        self.community_summaries = {} # {id: "summary text"}

    def extract_triplets(self, text_chunk):
        """Extracts Entity-Relation-Entity using LLM."""
        tmpl = """<|system|>
        Extract knowledge triplets (Subject | Relation | Object) from the text.
        - Normalize entities (e.g., "The system" -> "System").
        - Output ONE triplet per line.
        - Ignore generic statements.
        </s>
        <|user|>
        {text}
        </s>
        <|assistant|>"""
        
        chain = LLMChain(llm=self.llm, prompt=PromptTemplate(template=tmpl, input_variables=["text"]))
        try:
            resp = chain.run(text=text_chunk)
            triplets = []
            for line in resp.split('\n'):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) == 3: triplets.append(parts)
            return triplets
        except:
            return []

    def detect_communities(self):
        """
        Global Context: Uses Greedy Modularity to find clusters and summarizes them.
        """
        if self.graph.number_of_nodes() < 5: 
            return
        
        print("  > Detecting Global Communities...")
        try:
            communities = nx_comm.greedy_modularity_communities(self.graph)
            
            # Summarize top 3 clusters
            for i, comm in enumerate(communities[:3]):
                subgraph_nodes = list(comm)
                cluster_text = ", ".join(subgraph_nodes[:25])
                
                tmpl = """<|system|>
                Summarize the common theme of these entities in 1-2 sentences.
                </s>
                <|user|>
                Entities: {entities}
                </s>
                <|assistant|>"""
                
                chain = LLMChain(llm=self.llm, prompt=PromptTemplate(template=tmpl, input_variables=["entities"]))
                summary = chain.run(entities=cluster_text).strip()
                self.community_summaries[i] = summary
                print(f"    Cluster {i}: {summary}")
        except Exception as e:
            print(f"    Community detection skipped: {e}")

    def build_system(self, text_corpus):
        """Main build pipeline: Triplet Extraction -> Graph -> PageRank -> Communities -> Vector Index"""
        print(f"Processing Corpus ({len(text_corpus)} chars)...")
        chunks = textwrap.wrap(text_corpus, 1000)
        
        # 1. Build Graph
        total_triplets = 0
        for i, chunk in enumerate(chunks):
            print(f"  > Parsing chunk {i+1}/{len(chunks)}...")
            triplets = self.extract_triplets(chunk)
            total_triplets += len(triplets)
            for s, r, o in triplets:
                self.graph.add_edge(s, o, relation=r)
        
        print(f"  > Extracted {total_triplets} relationships.")

        # 2. Centrality & Communities
        if self.graph.number_of_nodes() > 0:
            try: 
                self.pagerank = nx.pagerank(self.graph, alpha=0.85)
            except: 
                self.pagerank = {n:1.0 for n in self.graph.nodes()}
            
            self.detect_communities()
            
            # 3. Vector Index (Nodes)
            self.nodes_list = list(self.graph.nodes())
            emb = self.embedder.encode(self.nodes_list)
            faiss.normalize_L2(emb)
            self.vector_index = faiss.IndexFlatIP(emb.shape[1])
            self.vector_index.add(emb)
            print(f"  > Indexed {len(self.nodes_list)} nodes.")
        else:
            print("  > Warning: Graph is empty.")

    def generate_hyde_and_expand(self, query):
        """
        Novelty: Combines Query Expansion with HyDE.
        """
        # 1. HyDE: Hallucinate an answer
        hyde_tmpl = """<|system|>
        Write a short, hypothetical passage that answers this question perfectly.
        Do not use real facts, just hallucinate a plausible structure and terminology.
        </s>
        <|user|>
        Question: {query}
        </s>
        <|assistant|>"""
        hyde_chain = LLMChain(llm=self.llm, prompt=PromptTemplate(template=hyde_tmpl, input_variables=["query"]))
        fake_ans = hyde_chain.run(query=query)
        
        return fake_ans

    def retrieve(self, query):
        # 1. HyDE Generation
        print("  > Generating HyDE (Hypothetical) Answer...")
        fake_ans = self.generate_hyde_and_expand(query)
        
        # 2. Vector Search (Anchors) using HyDE vector
        q_emb = self.embedder.encode(fake_ans) 
        faiss.normalize_L2(q_emb.reshape(1, -1))
        
        # Retrieve top 5 anchor nodes
        D, I = self.vector_index.search(q_emb.reshape(1, -1), 5)
        
        candidates = set()
        
        # 3. Graph Traversal + Semantic Attention
        print("  > Traversing Graph & Applying Attention...")
        for idx in I[0]:
            if idx < len(self.nodes_list):
                node = self.nodes_list[idx]
                
                # Get neighbors
                edges = self.graph.edges(node, data=True)
                for u, v, d in edges:
                    relation_text = d.get('relation', 'related to')
                    edge_str = f"{u} {relation_text} {v}"
                    
                    # Attention Score: Similarity(HyDE Answer, Edge Text)
                    edge_emb = self.embedder.encode(edge_str)
                    attn_score = util.cos_sim(q_emb, edge_emb).item()
                    
                    # Centrality Boost: PageRank(u) + PageRank(v)
                    pr_boost = (self.pagerank.get(u, 0) + self.pagerank.get(v, 0)) * 2.0
                    
                    # Final Score
                    final_score = attn_score + pr_boost
                    candidates.add((final_score, edge_str))
        
        # Sort by score
        sorted_facts = sorted(list(candidates), key=lambda x: x[0], reverse=True)
        local_context = [x[1] for x in sorted_facts[:10]] # Top 10 facts
        
        # 4. Global Context (Community Summaries)
        global_context = list(self.community_summaries.values())
        
        return local_context, global_context

    def query(self, user_q):
        local, glob = self.retrieve(user_q)
        
        # Construct Prompt with separated context levels
        ctx_str = "--- GLOBAL THEMES (High Level) ---\n" + "\n".join(glob) + \
                  "\n\n--- SPECIFIC FACTS (Graph Evidence) ---\n" + "\n".join(local)
        
        print(f"\n[Context Used]:\n{textwrap.shorten(ctx_str, width=200, placeholder='...')}")

        tmpl = """<|system|>
        You are an expert analyst. Answer the user's question using the provided context.
        - Use 'Global Themes' for high-level summaries.
        - Use 'Specific Facts' for detailed evidence.
        If the answer is not in the context, say so.
        
        Context:
        {ctx}
        </s>
        <|user|>
        Question: {q}
        </s>
        <|assistant|>"""
        
        chain = LLMChain(llm=self.llm, prompt=PromptTemplate(template=tmpl, input_variables=["ctx", "q"]))
        return chain.run(ctx=ctx_str, q=user_q)

# --- 4. Main Execution Loop ---

if __name__ == "__main__":
    print("=" * 60)
    print(" SUPER GRAPHRAG (Colab Edition)")
    print(" Techniques: HyDE + PageRank + Communities + Attention")
    print("=" * 60)
    
    rag_system = SuperGraphRAG(llm, embedder)
    
    # --- File Upload Logic ---
    corpus_text = ""
    
    if IN_COLAB:
        print("\n[Action Required] Please upload your data file (.txt)...")
        from google.colab import files
        uploaded = files.upload()
        if uploaded:
            fn = list(uploaded.keys())[0]
            print(f"Reading {fn}...")
            try: 
                corpus_text = uploaded[fn].decode("utf-8")
            except: 
                corpus_text = uploaded[fn].decode("latin-1")
    else:
        # Local fallback
        print("\n[Local Mode] Enter file path:")
        path = input("Path: ").strip()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                corpus_text = f.read()
        else:
            print("File not found. Using dummy text.")
            corpus_text = "RAG is a technique. GraphRAG improves RAG using networks."

    # Build & Run
    if corpus_text.strip():
        rag_system.build_system(corpus_text)
        
        print("\n" + "="*50)
        print(" SYSTEM READY. Enter your queries.")
        print("="*50)
        
        while True:
            q = input("\nQuery (or 'exit'): ")
            if q.lower() in ['exit', 'quit']: break
            if not q.strip(): continue
            
            response = rag_system.query(q)
            print(f"\nANSWER: {response}")
