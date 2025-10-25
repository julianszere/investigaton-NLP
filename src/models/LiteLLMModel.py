from litellm import completion


class LiteLLMModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def reply(self, messages):
        response = completion(model=self.model_name, messages=messages)
        return response.choices[0].message.content
