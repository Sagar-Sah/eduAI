from transformers import pipeline
from utils.rag_utils import get_similar_chunks
import re
from typing import Tuple, List

# Initialize model - using T5 for better instruction following
qa_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1,  # CPU
    max_length=512
)

def extract_relevant_segment(context: str, question: str) -> str:
    """
    Finds the most relevant segment within context for the question
    using keyword matching and proximity analysis
    """
    question_keywords = set(re.findall(r'\w+', question.lower()))
    
    best_segment = ""
    best_score = 0
    
    # Split context into meaningful segments
    segments = re.split(r'\n\n|\n(?=\w)', context)
    
    for segment in segments:
        segment_words = set(re.findall(r'\w+', segment.lower()))
        common_words = question_keywords & segment_words
        
        # Score based on keyword matches and segment length
        score = len(common_words) * 10 + len(segment.split()) / 10
        
        if score > best_score:
            best_score = score
            best_segment = segment
    
    return best_segment if best_score > 1 else context[:500]  # Fallback to first 500 chars

def generate_informed_response(question: str, context: str) -> str:
    """Generates answer using question and most relevant context segment"""
    prompt = f"""
    Question: {question}
    Context: {context}
    
    Requirements:
    - Answer truthfully using only the context
    - Be concise (1-2 sentences)
    - If unsure, say "I'm not certain but based on the information..."
    - Never invent information
    
    Answer:"""
    
    response = qa_model(
        prompt,
        max_new_tokens=150,
        num_beams=3,
        early_stopping=True,
        no_repeat_ngram_size=2
    )[0]['generated_text']
    
    # Post-processing
    answer = response.split("Answer:")[-1].strip()
    answer = re.sub(r'\s+', ' ', answer)  # Normalize whitespace
    
    # Ensure answer is not empty
    if not answer or len(answer.split()) < 3:
        return "I couldn't find a precise answer, but here's some relevant information: " + context[:200] + "..."
    
    return answer

def answer_question(question: str) -> Tuple[str, List[str]]:
    """Main QA pipeline with robust error handling"""
    try:
        # Validate input
        question = question.strip()
        if not question or len(question.split()) < 2:
            return "Please ask a more specific question", []
        
        # Retrieve context
        chunks = get_similar_chunks(question, top_k=5)
        if not chunks:
            return "I couldn't find any relevant information", []
        
        # Process context
        full_context = "\n\n".join(chunks)
        relevant_context = extract_relevant_segment(full_context, question)
        
        # Generate answer
        answer = generate_informed_response(question, relevant_context)
        
        return answer, chunks
    
    except Exception as e:
        print(f"Error processing question: {e}")
        return "An error occurred while processing your question", []

# from llama_cpp import Llama
# from utils.rag_utils import get_similar_chunks

# llm = Llama(model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=2048)

# def answer_question(query):
#     top_chunks = get_similar_chunks(query)
#     context = "\n".join(top_chunks)
#     prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
#     response = llm(prompt, max_tokens=150)['choices'][0]['text'].strip()
#     return response, top_chunks

# def get_reference_answer(question):
#     top_chunks = get_similar_chunks(question)
#     return " ".join(top_chunks[:1])
