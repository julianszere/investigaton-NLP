from abc import ABC, abstractmethod

class MemoryAgent(ABC):
    @abstractmethod
    def answer(self, sessions, question, t_question):
        pass