# -*- coding: utf-8 -*-
"""
GraphRAG Enterprise Assistant - Google Colab Executable
=======================================================
This script implements a Retrieval-Augmented Generation (RAG) system using
NetworkX for Knowledge Graph construction and a 4-bit quantized LLM for
reasoning.

INSTRUCTIONS FOR GOOGLE COLAB:
1. Copy this entire code into a code cell.
2. Ensure Runtime is set to GPU (Runtime > Change runtime type > T4 GPU).
3. Run the cell.
4. When prompted, upload your .txt data file.
"""

import os
import subprocess
import sys
import networkx as nx
import matplotlib.pyplot as plt
import torch
import textwrap

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
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    print("Dependencies installed.")

# Run installation if in Colab or missing deps
try:
    import google.colab
    IN_COLAB = True
    install_dependencies()
except ImportError:
    IN_COLAB = False
    pass

# --- 2. Model Loading (4-bit Quantization for Colab T4) ---
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

print("\nLoading LLM (Zephyr-7b-beta)...")

# Configuration for 4-bit quantization to fit in free Colab GPU RAM
bnb_config = None
try:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
except ImportError:
    print("Warning: bitsandbytes not found, loading standard model (might OOM).")

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
    print(f"Error loading model: {e}")
    sys.exit(1)

# Create Text Generation Pipeline
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# --- 3. Knowledge Graph Engine ---

class KnowledgeGraphRAG:
    def __init__(self, llm_engine):
        self.llm = llm_engine
        self.graph = nx.DiGraph()
        
    def extract_triplets(self, text_chunk):
        """
        Uses LLM to parse text into (Subject, Relation, Object) triplets.
        """
        prompt_template = """
        <|system|>
        You are a data processing expert. Extract knowledge graph triplets from the text below.
        Format: Subject | Relation | Object
        Ignore vague statements. Focus on entities and specific actions.
        Output ONLY the triplets, one per line.
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
        Ingests text, extracts triplets, and builds the NetworkX graph.
        """
        print(f"Extracting entities from corpus ({len(text_corpus)} chars)...")
        # Split text strictly for processing chunks
        chunks = textwrap.wrap(text_corpus, 1000) 
        
        all_triplets = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            triplets = self.extract_triplets(chunk)
            all_triplets.extend(triplets)
            
        print(f"Found {len(all_triplets)} relationships. Building Graph...")
        
        for subj, rel, obj in all_triplets:
            self.graph.add_edge(subj, obj, relation=rel)
            
    def visualize(self):
        """Simple visualization of the Knowledge Graph"""
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty. No entities found.")
            return

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, k=0.5)
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', 
                node_size=1500, edge_color='gray', font_size=8, font_weight='bold')
        
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=6)
        plt.title("Extracted Knowledge Graph")
        plt.show()

    def retrieve_context(self, query):
        """
        Traverses the graph to find nodes relevant to the query.
        Simple fuzzy matching for this demo.
        """
        query_terms = query.lower().split()
        relevant_triplets = []
        
        for node in self.graph.nodes():
            # Check if node matches query terms
            if any(term in node.lower() for term in query_terms):
                # Get neighbors (1-hop)
                out_edges = self.graph.out_edges(node, data=True)
                in_edges = self.graph.in_edges(node, data=True)
                
                for u, v, data in out_edges:
                    relevant_triplets.append(f"{u} {data['relation']} {v}")
                for u, v, data in in_edges:
                    relevant_triplets.append(f"{u} {data['relation']} {v}")
                    
        # Deduplicate
        return list(set(relevant_triplets))

    def answer_query(self, query):
        context = self.retrieve_context(query)
        context_str = "\n".join(context) if context else "No direct graph connections found."
        
        print(f"\n[Graph Context Retrieved]:\n{context_str}\n")
        
        prompt_template = """
        <|system|>
        You are an intelligent enterprise assistant. Use the provided Knowledge Graph Context to answer the user's question. 
        If the answer is not in the context, say so.
        </s>
        <|user|>
        Context:
        {context}
        
        Question: {query}
        </s>
        <|assistant|>
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(context=context_str, query=query)

# --- 4. Main Execution Loop ---

if __name__ == "__main__":
    print("-" * 50)
    print("GraphRAG System Initialization")
    print("-" * 50)
    
    rag_system = KnowledgeGraphRAG(llm)
    
    # --- File Upload Logic ---
    corpus_text = ""
    
    if IN_COLAB:
        print("\n[Action Required] Please upload your text file (.txt)...")
        from google.colab import files
        uploaded = files.upload()
        
        if not uploaded:
            print("No file uploaded. Using default demo text.")
            # Default fallback
            corpus_text = "RAG is a technique to retrieve data. LoRA is for fine-tuning."
        else:
            # Read the first uploaded file
            filename = list(uploaded.keys())[0]
            print(f"Reading {filename}...")
            try:
                corpus_text = uploaded[filename].decode("utf-8")
            except Exception:
                 # Fallback for non-utf8
                corpus_text = uploaded[filename].decode("latin-1")
    else:
        # Local execution fallback
        print("\n[Local Mode] Enter path to your text file:")
        file_path = input("Path: ").strip()
        if os.path.exists(file_path):
             with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                corpus_text = f.read()
        else:
            print("File not found. Using minimal demo text.")
            corpus_text = "RAG is a technique to retrieve data. LoRA is for fine-tuning."

    # Build Graph
    if corpus_text.strip():
        rag_system.build_graph(corpus_text)
        
        # Visualize
        try:
            rag_system.visualize()
        except Exception as e:
            print("Visualization skipped.")
            
        # Interactive Query Loop
        print("\n" + "="*50)
        print(" SYSTEM READY. Enter your queries based on the uploaded file.")
        print(" (Type 'exit' to quit)")
        print("="*50)
        
        while True:
            user_input = input("\nEnter query: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            
            if not user_input.strip():
                continue
            
            response = rag_system.answer_query(user_input)
            print(f"Answer: {response}")
    else:
        print("Error: No text data provided to build graph.")
