from transformers import pipeline
import random
import re

# Initialize pipelines
qg_pipeline = pipeline(
    "text2text-generation",
    model="valhalla/t5-base-qg-hl",
    device=-1  # CPU
)

answer_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1  # CPU
)

def clean_text(text):
    """Clean and normalize text for processing"""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
    text = text.strip()
    return text[:2000]  # Limit input size

def generate_question(text):
    """Generate a single question from text"""
    prompt = f"Generate a question about: {text}"
    generated = qg_pipeline(
        prompt,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,  # More creative questions
        do_sample=True
    )
    if not generated:
        return None
    question = generated[0]['generated_text'].split("?")[0] + "?"
    return clean_text(question)

def generate_answer(question, context):
    """Generate accurate answer using context"""
    prompt = f"""Answer this question based on the context:
    
    Question: {question}
    Context: {context}
    
    Provide a concise answer (3-5 words):"""
    
    answer = answer_pipeline(
        prompt,
        max_length=50,
        num_return_sequences=1,
        temperature=0.3,  # More factual
        do_sample=False  # Greedy decoding
    )[0]['generated_text']
    return clean_text(answer)

def generate_distractors(question, correct_answer, context):
    """Generate plausible wrong answers"""
    distractors = []
    
    # Strategy 1: Similar but incorrect
    prompt1 = f"""Generate a wrong but plausible answer for:
    Question: {question}
    Correct Answer: {correct_answer}
    Context: {context}
    
    Wrong Answer:"""
    
    # Strategy 2: Opposite meaning
    prompt2 = f"""Generate an opposite answer for:
    Question: {question}
    Correct Answer: {correct_answer}
    
    Opposite Answer:"""
    
    # Strategy 3: Random fact from context
    prompt3 = f"""Extract a random but wrong fact from this context for:
    Question: {question}
    Context: {context}
    
    Wrong Fact:"""
    
    for prompt in [prompt1, prompt2, prompt3]:
        distractor = answer_pipeline(
            prompt,
            max_length=40,
            num_return_sequences=1,
            temperature=0.9,  # More creative
            do_sample=True
        )[0]['generated_text']
        distractor = clean_text(distractor)
        if distractor.lower() != correct_answer.lower():
            distractors.append(distractor)
    
    return list(set(distractors))[:3]  # Remove duplicates

def generate_mcqs(text, num_questions=3):
    """Generate quality multiple choice questions"""
    text = clean_text(text)
    questions = []
    
    for _ in range(num_questions):
        try:
            # Generate question
            question = generate_question(text)
            if not question:
                continue
                
            # Generate correct answer
            answer = generate_answer(question, text)
            if not answer:
                continue
                
            # Generate distractors
            distractors = generate_distractors(question, answer, text)
            if len(distractors) < 2:  # Need at least 2 distractors
                continue
                
            # Prepare options
            options = [answer] + distractors[:3]
            random.shuffle(options)
            
            questions.append({
                "question": question,
                "options": options,
                "answer": answer,
                "context_snippet": text[:200] + "..." if len(text) > 200 else text
            })
            
        except Exception as e:
            print(f"Error generating MCQ: {str(e)}")
            continue
    
    return questions

