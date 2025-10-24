from src.agents.MemoryAgent import MemoryAgent


class FullContextAgent(MemoryAgent):
    def __init__(self, model="azure/gpt-4.1"):
        self.model = model

    def answer(self, sessions, question, t_question):
        evidence = ""
        for session in sessions:
            for message in session:
                evidence += f"{message['role']}: {message['content']}\n"

        prompt = f"""
        You are a helpful assistant that answers a question based on the evidence.
        The evidence is: {evidence}
        The question is: {question}
        Return the answer to the question.
        """
        messages = [{"role": "user", "content": prompt}]
        answer = self.model.reply(messages)
        return answer
