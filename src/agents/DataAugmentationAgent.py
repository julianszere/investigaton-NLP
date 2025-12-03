from src.datasets.LongMemEvalDataset import LongMemEvalInstance

class FullConversationQAGenerator:
    def __init__(self, model):
        self.model = model

    def answer(self, session):
        """
        session should be a single session object with a .messages list:
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
        """

        # Build full conversation text from the single session
        conversation_lines = []
        for msg in session.messages:
            role = msg["role"]
            content = msg["content"]
            conversation_lines.append(f"{role.upper()}: {content}")

        conversation_text = "\n".join(conversation_lines)

        prompt = f"""
        You are an assistant that analyzes a **full conversation** between a user and an assistant.

        ⚠️ IMPORTANT INSTRUCTION:
        The **summary and the Q&A you produce will be used as memory data** for a long-term memory agent.
        Therefore:
        - The summary must be **deep, comprehensive, and information-dense**, capturing all relevant facts, events, preferences, intentions, tasks, and dependencies.
        - The questions must cover **every important detail**, including:
        • local details  
        • long-range dependencies across distant parts of the conversation  
        • implicit reasoning  
        • hidden constraints  
        • meta-information about the user (goals, plans, preferences, issues faced)  
        - The answers must be **strictly correct**, precise, and grounded in the conversation.

        TASK:
        1. Produce a **detailed but concise (max 250 words)** summary capturing all relevant information that should be memorized.
        2. Generate **N high-quality questions** that fully probe the important content of the conversation. You can decide the number N based on the amount of information available to QA about.
        - Some must require combining information from distant messages.
        - Include factual, reasoning-heavy, and meta-level questions.
        3. Provide **the correct answer** to each question, grounded only in the conversation.

        FORMAT:
        SUMMARY:
        <your summary here>

        Q&A:
        1. Q: <question 1>
        A: <answer 1>

        2. Q: <question 2>
        A: <answer 2>

        ...

        FULL CONVERSATION:
        {conversation_text}
        """

        messages = [{"role": "user", "content": prompt}]
        return self.model.reply(messages)