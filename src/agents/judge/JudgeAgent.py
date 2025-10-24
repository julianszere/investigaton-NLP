class JudgeAgent:
    def __init__(self, model):
        self.model = model

    def judge(self, question, memory_agent_answer, ground_truth_answer):
        prompt = f"""
        You are a helpful assistant that judges the correctness of an answer to a question.
        The question is: {question}
        The memory agent answer is: {memory_agent_answer}
        The ground truth answer is: {ground_truth_answer}
        Return True if the prediction is correct, False otherwise. No other text or explanation.
        """
        messages = [{"role": "user", "content": prompt}]
        judgment = self.model.reply(messages)
        return eval(judgment)
