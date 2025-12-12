# -*- coding: utf-8 -*-
"""
Advanced GraphRAG Enterprise Assistant - Google Colab Executable
================================================================
This script implements an Advanced Hybrid RAG system combining:
1. Knowledge Graph (NetworkX) for structural reasoning.
2. Vector Database (FAISS) for semantic entry points.
3. Query Expansion (LLM) for broader recall.
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
        "faiss-gpu"
    ]
    for package in packages:
        try:
            __import__(package.replace("-", "_").split("==")[0])
        except ImportError:
            # Handle package name discrepancies
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
from sentence_transformers import SentenceTransformer
import faiss

print("\nLoading Models...")

# A. Load Embedding Model for Vector Search
# Lightweight model, fast and effective for this scale
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
    temperature=0.1, # Low temp for factual consistency
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
        self.nodes_list = [] # Maps index ID to node name
        
    def extract_triplets(self, text_chunk):
        """
        Uses LLM to parse text into (Subject, Relation, Object) triplets.
        Enhanced system prompt for better stability.
        """
        prompt_template = """
        <|system|>
        You are a Knowledge Graph Expert. Extract structured relationships from the text.
        Return ONLY triplets in this format: Subject | Relation | Object
        Rules:
        1. Subjects and Objects must be specific entities (nouns).
        2. Relations should be verbs or short phrases.
        3. Ignore generic or vague sentences.
        4. One triplet per line.
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
        1. Extract Triplets -> Build NetworkX Graph
        2. Embed Node Names -> Build FAISS Vector Index
        """
        print(f"Processing corpus ({len(text_corpus)} chars)...")
        chunks = textwrap.wrap(text_corpus, 1000) 
        
        all_triplets = []
        for i, chunk in enumerate(chunks):
            print(f"  > Analyzing chunk {i+1}/{len(chunks)}...")
            triplets = self.extract_triplets(chunk)
            all_triplets.extend(triplets)
            
        print(f"Found {len(all_triplets)} relationships. Constructing Graph & Indices...")
        
        # 1. Build Graph
        for subj, rel, obj in all_triplets:
            self.graph.add_edge(subj, obj, relation=rel)
            
        # 2. Build Vector Index for Hybrid Search
        if self.graph.number_of_nodes() > 0:
            self.nodes_list = list(self.graph.nodes())
            embeddings = self.embedder.encode(self.nodes_list)
            
            # Normalize for cosine similarity (FAISS uses L2 by default, normalization makes it cosine)
            faiss.normalize_L2(embeddings)
            
            dimension = embeddings.shape[1]
            self.vector_index = faiss.IndexFlatIP(dimension) # Inner Product (Cosine if normalized)
            self.vector_index.add(embeddings)
            print(f"Indexed {len(self.nodes_list)} nodes in FAISS.")
        else:
            print("Warning: Graph is empty.")

    def expand_query(self, query):
        """
        Advanced Step 1: Query Expansion
        Generates variations of the user query to increase recall.
        """
        prompt_template = """
        <|system|>
        You are a helpful AI assistant. Generate 2 alternative versions of the user's question to help find relevant information. 
        Focus on synonyms and related concepts. Output ONLY the questions, one per line.
        </s>
        <|user|>
        Original Question: {query}
        </s>
        <|assistant|>
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["query"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        variations = chain.run(query=query).strip().split('\n')
        # Clean list
        variations = [v.strip() for v in variations if v.strip()]
        return [query] + variations[:2] # Original + up to 2 variations

    def hybrid_retrieval(self, queries, top_k=3):
        """
        Advanced Step 2: Hybrid Search (Vector + Graph)
        1. Search vector index for best matching nodes (Entry Points).
        2. Traverse graph from those nodes to get context (Neighbors).
        """
        relevant_context = set()
        
        print(f"  > Retrieving for queries: {queries}")
        
        # Vector Search for Entry Points
        for q in queries:
            q_embed = self.embedder.encode([q])
            faiss.normalize_L2(q_embed)
            
            # Search FAISS
            if self.vector_index:
                D, I = self.vector_index.search(q_embed, top_k)
                for idx in I[0]:
                    if idx < len(self.nodes_list):
                        node_name = self.nodes_list[idx]
                        
                        # Graph Traversal (1-hop)
                        out_edges = self.graph.out_edges(node_name, data=True)
                        in_edges = self.graph.in_edges(node_name, data=True)
                        
                        for u, v, data in out_edges:
                            relevant_context.add(f"{u} --[{data['relation']}]--> {v}")
                        for u, v, data in in_edges:
                            relevant_context.add(f"{u} --[{data['relation']}]--> {v}")
                            
        return list(relevant_context)

    def answer_query(self, query):
        # 1. Expand Query
        expanded_queries = self.expand_query(query)
        
        # 2. Hybrid Retrieval
        context_list = self.hybrid_retrieval(expanded_queries)
        context_str = "\n".join(context_list) if context_list else "No direct info found in Knowledge Graph."
        
        print(f"\n[Retrieved Context ({len(context_list)} facts)]:\n{textwrap.shorten(context_str, width=300, placeholder='...')}\n")
        
        # 3. Generation
        prompt_template = """
        <|system|>
        You are an expert assistant. Answer the question based strictly on the provided Knowledge Graph Context.
        
        Context:
        {context}
        
        If the answer isn't in the context, state that you don't have enough information.
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
        """Visualizes the graph"""
        if self.graph.number_of_nodes() == 0:
            print("Graph empty.")
            return

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(self.graph, k=0.3, iterations=50)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightgreen', 
                node_size=1000, edge_color='gray', alpha=0.7, font_size=8)
        
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        # Draw only some labels to avoid clutter if huge
        if len(edge_labels) < 50:
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=6)
        
        plt.title("Knowledge Graph")
        plt.show()

# --- 4. Main Execution ---

if __name__ == "__main__":
    print("=" * 60)
    print("Advanced Hybrid GraphRAG System (Colab T4 Edition)")
    print("=" * 60)
    
    rag_system = AdvancedGraphRAG(llm, embedder)
    
    # --- File Upload ---
    corpus_text = ""
    
    if IN_COLAB:
        print("\n[Action Required] Please upload your text file (.txt)...")
        from google.colab import files
        uploaded = files.upload()
        
        if uploaded:
            filename = list(uploaded.keys())[0]
            print(f"Reading {filename}...")
            try:
                corpus_text = uploaded[filename].decode("utf-8")
            except Exception:
                corpus_text = uploaded[filename].decode("latin-1")
        else:
            print("No file? Using demo text.")
            corpus_text = "GraphRAG combines knowledge graphs and vector search. It improves retrieval accuracy."
    else:
        # Local fallback
        print("\n[Local Mode] Paste path or text:")
        inp = input("Path or Text: ").strip()
        if os.path.exists(inp):
            with open(inp, 'r', encoding='utf-8', errors='ignore') as f:
                corpus_text = f.read()
        else:
            corpus_text = inp if inp else "GraphRAG uses vectors and graphs."

    # Build System
    if corpus_text.strip():
        rag_system.build_graph(corpus_text)
        
        try:
            rag_system.visualize()
        except:
            pass
            
        print("\n" + "="*50)
        print(" SYSTEM READY. (Using Hybrid Vector+Graph Search)")
        print("="*50)
        
        while True:
            user_input = input("\nEnter query (or 'exit'): ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip(): continue
            
            response = rag_system.answer_query(user_input)
            print(f"Answer: {response}")
