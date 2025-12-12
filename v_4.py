# -*- coding: utf-8 -*-
"""
Advanced GraphRAG Enterprise Assistant - Google Colab Executable
================================================================
Feature Set:
1. Knowledge Graph (NetworkX) with PageRank Centrality.
2. Semantic Graph Attention (Query-to-Edge Attention Mechanism).
3. Hybrid Retrieval (Vector Anchors + Attention Ranking).
4. 4-bit Quantized LLM (Zephyr-7b) for inference.

INSTRUCTIONS FOR GOOGLE COLAB:
1. Copy this entire code into a code cell.
2. Ensure Runtime is set to GPU (Runtime > Change runtime type > T4 GPU).
3. Run the cell.
4. Upload your .txt data file when prompted.
"""

import os
import subprocess
import sys
import networkx as nx
import matplotlib.pyplot as plt
import torch
import textwrap
import numpy as np

# --- 1. Environment Setup & Dependency Installation ---
def install_dependencies():
    print("Checking and installing dependencies... (This may take 2-3 minutes)")
    packages = [
        "transformers",
        "accelerate",
        "bitsandbytes",
        "langchain",
        "langchain-community",
        "networkx",
        "sentence-transformers",
        "faiss-gpu",
        "scipy" # Required for PageRank
    ]
    for package in packages:
        try:
            __import__(package.replace("-", "_").split("==")[0])
        except ImportError:
            pkg_name = package
            if package == "faiss-gpu": pkg_name = "faiss-gpu"
            if package == "sentence-transformers": pkg_name = "sentence-transformers"
            print(f"Installing {pkg_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name, "-q"])
    print("Dependencies installed.")

try:
    import google.colab
    IN_COLAB = True
    install_dependencies()
except ImportError:
    IN_COLAB = False
    pass

# --- 2. Imports & Model Loading ---
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer, util
import faiss

print("\nLoading Models...")

# A. Load Embedding Model (Attention Head)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# B. Load LLM (Zephyr-7b-beta) with 4-bit Quantization
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
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
except Exception as e:
    print(f"Error loading LLM: {e}")
    sys.exit(1)

text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.01, # Extremely low for strict adherence to context
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# --- 3. Advanced Knowledge Graph Engine ---

class AdvancedGraphRAG:
    def __init__(self, llm_engine, embedder_model):
        self.llm = llm_engine
        self.embedder = embedder_model
        self.graph = nx.DiGraph()
        self.vector_index = None
        self.nodes_list = [] 
        self.pagerank_scores = {}
        
    def extract_triplets(self, text_chunk):
        """
        Extracts structured knowledge.
        """
        prompt_template = """
        <|system|>
        Extract knowledge triplets (Subject | Relation | Object) from the text.
        - Normalize entities (e.g., "The project" -> "Project").
        - Capture actions and properties.
        - Output ONLY format: Entity1 | Relation | Entity2
        </s>
        <|user|>
        Text: {text}
        </s>
        <|assistant|>
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(text=text_chunk)
        
        triplets = []
        for line in response.split('\n'):
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) == 3:
                    triplets.append((parts[0], parts[1], parts[2]))
        return triplets

    def build_graph(self, text_corpus):
        """
        Builds graph and computes Centrality (PageRank).
        """
        print(f"Processing corpus ({len(text_corpus)} chars)...")
        chunks = textwrap.wrap(text_corpus, 1000) 
        
        all_triplets = []
        for i, chunk in enumerate(chunks):
            print(f"  > Analyzing chunk {i+1}/{len(chunks)}...")
            triplets = self.extract_triplets(chunk)
            all_triplets.extend(triplets)
            
        print(f"Found {len(all_triplets)} relationships. Constructing Graph...")
        
        # 1. Build Graph
        for subj, rel, obj in all_triplets:
            self.graph.add_edge(subj, obj, relation=rel)
            
        # 2. Compute Centrality (PageRank)
        if self.graph.number_of_nodes() > 0:
            try:
                self.pagerank_scores = nx.pagerank(self.graph, alpha=0.85)
                print("  > Centrality (PageRank) computed.")
            except Exception as e:
                print(f"  > PageRank failed (likely sparse graph), using uniform weights.")
                self.pagerank_scores = {n: 1.0 for n in self.graph.nodes()}

            # 3. Build Vector Index
            self.nodes_list = list(self.graph.nodes())
            embeddings = self.embedder.encode(self.nodes_list)
            faiss.normalize_L2(embeddings)
            
            dimension = embeddings.shape[1]
            self.vector_index = faiss.IndexFlatIP(dimension)
            self.vector_index.add(embeddings)
            print(f"Indexed {len(self.nodes_list)} nodes in FAISS.")
        else:
            print("Warning: Graph is empty.")

    def compute_attention(self, query_emb, triplet_text):
        """
        Novel Mechanism: Graph Attention
        Computes semantic similarity between Query and a specific Graph Edge (Triplet).
        """
        triplet_emb = self.embedder.encode(triplet_text)
        # Cosine similarity
        score = util.cos_sim(query_emb, triplet_emb).item()
        return score

    def retrieve_with_attention(self, query, top_k_anchors=3, top_n_triplets=10):
        """
        Hybrid Retrieval with Attention-Based Reranking
        """
        query_emb = self.embedder.encode(query)
        
        # 1. Vector Search for Anchors (Entry Points)
        faiss_query = np.array([query_emb]).astype('float32')
        faiss.normalize_L2(faiss_query)
        D, I = self.vector_index.search(faiss_query, top_k_anchors)
        
        candidate_triplets = set()
        
        # 2. Graph Traversal (Expansion)
        for idx in I[0]:
            if idx < len(self.nodes_list):
                anchor = self.nodes_list[idx]
                
                # Get both outgoing and incoming edges
                edges = list(self.graph.out_edges(anchor, data=True)) + \
                        list(self.graph.in_edges(anchor, data=True))
                
                for u, v, data in edges:
                    triplet_str = f"{u} {data['relation']} {v}"
                    candidate_triplets.add(triplet_str)

        # 3. Apply Attention Mechanism (Re-ranking)
        scored_triplets = []
        for triplet in candidate_triplets:
            # S_attn: Semantic relevance of the specific fact to the query
            attn_score = self.compute_attention(query_emb, triplet)
            
            # S_centrality: Importance of the entities involved (average PageRank)
            # This boosts "main characters" or "core concepts"
            parts = triplet.split(' ', 1) # simple split to guess subject
            subj = parts[0]
            
            # Safe get for pagerank (default to small value if not found)
            pr_score = self.pagerank_scores.get(subj, 0.001)
            
            # Final Score: 70% Attention (Relevance), 30% Centrality (Importance)
            final_score = (attn_score * 0.7) + (pr_score * 5.0 * 0.3) # Scaling PR up
            
            scored_triplets.append((final_score, triplet))
            
        # Sort by score and take top N
        scored_triplets.sort(key=lambda x: x[0], reverse=True)
        final_context = [t[1] for t in scored_triplets[:top_n_triplets]]
        
        return final_context

    def answer_query(self, query):
        context_list = self.retrieve_with_attention(query)
        context_str = "\n".join(context_list) if context_list else "No relevant context found."
        
        print(f"\n[Graph Attention Context]:\n{textwrap.shorten(context_str, width=200, placeholder='...')}\n")
        
        prompt_template = """
        <|system|>
        Answer the user's question using ONLY the context provided.
        Context is ranked by semantic attention.
        
        Context:
        {context}
        </s>
        <|user|>
        Question: {query}
        </s>
        <|assistant|>
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(context=context_str, query=query)

    def visualize(self):
        """Visualizes the graph with node size based on PageRank"""
        if self.graph.number_of_nodes() == 0: return

        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(self.graph, k=0.4)
        
        # Scale node size by PageRank
        node_sizes = [self.pagerank_scores.get(n, 0.01) * 5000 + 300 for n in self.graph.nodes()]
        
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', 
                node_size=node_sizes, edge_color='#888888', alpha=0.8, font_size=8)
        
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        if len(edge_labels) < 40:
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=6)
        
        plt.title("Graph w/ PageRank Centrality")
        plt.show()

# --- 4. Main Execution ---

if __name__ == "__main__":
    print("=" * 60)
    print("Novel GraphRAG: Attention & Centrality (Colab T4)")
    print("=" * 60)
    
    rag_system = AdvancedGraphRAG(llm, embedder)
    
    # --- File Upload ---
    corpus_text = ""
    
    if IN_COLAB:
        print("\n[Action Required] Please upload your text file (.txt)...")
        from google.colab import files
        uploaded = files.upload()
        if uploaded:
            fn = list(uploaded.keys())[0]
            try: corpus_text = uploaded[fn].decode("utf-8")
            except: corpus_text = uploaded[fn].decode("latin-1")
    else:
        # Local fallback
        print("\n[Local Mode] Paste path or text:")
        inp = input("Path or Text: ").strip()
        if os.path.exists(inp):
            with open(inp, 'r', encoding='utf-8', errors='ignore') as f:
                corpus_text = f.read()
        else:
            corpus_text = inp if inp else "GraphRAG with attention mechanisms."

    if corpus_text.strip():
        rag_system.build_graph(corpus_text)
        try: rag_system.visualize()
        except: pass
            
        print("\n" + "="*50)
        print(" SYSTEM READY. (Attention + PageRank Active)")
        print("="*50)
        
        while True:
            user_input = input("\nEnter query (or 'exit'): ")
            if user_input.lower() in ['exit', 'quit']: break
            if not user_input.strip(): continue
            response = rag_system.answer_query(user_input)
            print(f"Answer: {response}")
