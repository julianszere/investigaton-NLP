import os
from dotenv import load_dotenv
from litellm import completion
from utils import load_longmemeval_o_df

# Load environment variables from .env file
load_dotenv()


def llm_judge_answer(question, predicted_answer, answer, model="azure/gpt-4.1"):
    prompt = f"""
    You are a helpful assistant that judges the correctness of an answer to a question.
    The question is: {question}
    The prediction is: {predicted_answer}
    The answer is: {answer}
    Return True if the prediction is correct, False otherwise. No other text or explanation.
    """
    response = completion(model=model, messages=[{"role": "user", "content": prompt}])
    return eval(response.choices[0].message.content)


def llm_predict_answer(question, evidence, model="azure/gpt-4.1"):
    prompt = f"""
    You are a helpful assistant that answers a question based on the evidence.
    The question is: {question}
    The evidence is: {evidence}
    Return the answer to the question.
    """
    response = completion(model=model, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

N = 10
correct_predictions = 0
longmemeval_o_df = load_longmemeval_o_df().head(N)

for index, row in longmemeval_o_df.iterrows():
    question = row["question"]
    answer = row["answer"]

    sessions = row["haystack_sessions"]
    evidence = ""
    for session in sessions:
        for message in session:
            evidence += f"{message['role']}: {message['content']}\n"

    predicted_answer = llm_predict_answer(question, evidence)
    answer_is_correct = llm_judge_answer(question, predicted_answer, answer)
    if answer_is_correct:
        correct_predictions += 1

    print(f"Question: {question}")
    print(f"Predicted Answer: {predicted_answer}")
    print(f"Answer: {answer}")
    print(f"Correct: {answer_is_correct}")
    print("-" * 100)

print(f"Correct predictions: {correct_predictions}/{N}")
print(f"Accuracy: {correct_predictions/N}")