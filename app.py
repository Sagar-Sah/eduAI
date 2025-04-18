# app.py
from flask import Flask, request, jsonify, render_template
import os
from utils import pdf_utils, rag_utils, llm_utils, summarizer, question_gen, quiz_evaluator

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    file = request.files['pdf']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    raw_text = pdf_utils.extract_text(filepath)
    chunks = rag_utils.chunk_text(raw_text)
    rag_utils.store_embeddings(chunks)
    return jsonify({"message": "PDF processed successfully!"})

@app.route('/summarize', methods=['GET'])
def summarize():
    text = rag_utils.get_full_text()
    summary = summarizer.generate_summary(text)
    return jsonify({"summary": summary})

@app.route('/generate_questions', methods=['GET'])
def generate_questions():
    text = rag_utils.get_full_text()
    questions = question_gen.generate_mcqs(text)
    return jsonify({"questions": questions})

@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.json['question']
        if not question or len(question.strip()) < 3:
            return jsonify({"error": "Question too short"}), 400
            
        answer, chunks = llm_utils.answer_question(question)
        return jsonify({
            "answer": answer,
            "chunks": chunks[:3]  # Return first 3 chunks for reference
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Failed to process question"
        }), 500

@app.route('/quiz', methods=['POST'])
def quiz():
    try:
        question = request.json['question']
        user_answer = request.json['answer']
        reference = llm_utils.get_reference_answer(question)
        score, feedback = quiz_evaluator.evaluate(user_answer, reference)
        return jsonify({
            "score": score,
            "feedback": feedback,
            "correct_answer": reference[:200]  # Return partial correct answer
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Failed to evaluate quiz answer"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)