from litellm import completion
from src.models.Model import Model

class LiteLLMModel(Model):
    def __init__(self, model_name):
        super().__init__(model_name)

    def reply(self, messages, tools=None):
        response = completion(model=self.model_name, messages=messages)
        return response.choices[0].message.content
