# rerank BAAI/bge-reranker-v2-gemma

from colorama import Fore
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "BAAI/bge-reranker-v2-gemma"
INSTRUCTION = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
token_true_id = tokenizer.convert_tokens_to_ids("Yes")
token_false_id = tokenizer.convert_tokens_to_ids("No")
max_length = 8192

# Format the instruction with query and document
def format_instruction(query, doc, instruction=INSTRUCTION):
    return f"A: {query}\nB: {doc}\n{instruction}"

# Process inputs for the model
def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length
    )
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

# Compute scores for a query and a list of documents
@torch.no_grad()
def compute_scores(query, docs):
    pairs = [format_instruction(query, doc) for doc in docs]
    
    inputs = process_inputs(pairs)
    logits = model(**inputs).logits[:, -1, :]
    
    # Return raw scores for Yes token
    scores = logits[:, token_true_id]
    return scores.tolist()

def rerank_documents(query, docs):
    """
    Rerank documents based on relevance to query
    
    Args:
        query (str): The search query
        docs (list): List of document strings
        return_scores (bool): Whether to return scores along with documents
    
    Returns:
        list: Reranked documents (and scores if return_scores=True)
    """
    scores = compute_scores(query, docs)
    
    # Create list of (doc, score) pairs and sort by score descending
    doc_score_pairs = list(zip(docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    return doc_score_pairs

# Test
if __name__ == "__main__":
    query = "อะไรคือเมืองหลวงของประเทศไทย?"
    docs = [
        "ประเทศไทยตั้งอยู่ในเอเชียตะวันออกเฉียงใต้",
        "กรุงเทพฯ เป็นศูนย์กลางการเมืองและเศรษฐกิจของไทย",
        "กรุงเทพมหานครคือเมืองหลวงของประเทศไทย",
        "เชียงใหม่เป็นเมืองสำคัญในภาคเหนือของไทย",
        "หมูป่าพาหมาป่ามากินหมี่ที่เยาวราช"
    ]    
    
    print("Reranked documents:")
    reranked_docs = rerank_documents(query, docs)
    for i, (doc, score) in enumerate(reranked_docs):
        print(f"{i+1}. Score: {score:.4f} - Document: {doc}")
        
    print(Fore.BLUE + "="*50 + Fore.RESET)
