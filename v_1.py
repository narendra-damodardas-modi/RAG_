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
    # Assume local user has deps or will install them manually
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
        for chunk in chunks:
            triplets = self.extract_triplets(chunk)
            all_triplets.extend(triplets)
            
        print(f"Found {len(all_triplets)} relationships. Building Graph...")
        
        for subj, rel, obj in all_triplets:
            self.graph.add_edge(subj, obj, relation=rel)
            
    def visualize(self):
        """Simple visualization of the Knowledge Graph"""
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


# --- 4. Execution Data (From Abstract) ---

# Text from your uploaded abstract
corpus_text = """
Enhancing Conversational Enterprise/Personal Assistants through LoRA Fine-Tuning and Retrieval-Augmented Generation.
Advancements in large language models (LLMs) and multimodal AI systems have opened new frontiers for building intelligent conversational enterprise and personal assistants. 
However, delivering responses that are both context-aware and user-personalized remains a key challenge. 
Our project explores the synergistic application of Low-Rank Adaptation (LoRA) fine-tuning and Retrieval-Augmented Generation (RAG) to enhance the performance and adaptability of conversational assistants.
LoRA is a parameter-efficient fine-tuning technique that enables lightweight adaptation of large models to specific domains.
LoRA allows rapid customization of both text-based and image-generating models like Stable Diffusion.
RAG enhances dialogue systems by incorporating external context retrieval into the generation pipeline.
RAG retrieves relevant documents, image prompts, or style references from structured or unstructured sources.
When applied to image generation, RAG improves prompt specificity and semantic relevance.
The fusion of LoRA and RAG represents a scalable and cost-effective pathway toward the next generation of conversational agents.
"""

# --- 5. Main Execution Loop ---

if __name__ == "__main__":
    print("-" * 50)
    print("Initializing GraphRAG System...")
    rag_system = KnowledgeGraphRAG(llm)
    
    # Build Graph
    rag_system.build_graph(corpus_text)
    
    # Visualize (works best in Jupyter/Colab)
    try:
        rag_system.visualize()
    except Exception as e:
        print("Visualization skipped (requires display environment).")

    # Interactive Query Loop
    print("\n" + "="*50)
    print(" SYSTEM READY. Ask questions about the project.")
    print(" (Type 'exit' to quit)")
    print("="*50)
    
    # Pre-canned examples for immediate output demonstration
    example_queries = [
        "What is LoRA used for?",
        "How does RAG help image generation?",
        "What is the goal of the project?"
    ]
    
    print("\n--- Running Example Queries ---")
    for q in example_queries:
        print(f"\nQ: {q}")
        response = rag_system.answer_query(q)
        print(f"A: {response}")
        
    # Manual loop
    while True:
        user_input = input("\nEnter query: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        response = rag_system.answer_query(user_input)
        print(f"Answer: {response}")
