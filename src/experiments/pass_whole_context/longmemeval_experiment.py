from src.models.LiteLLMModel import LiteLLMModel
from src.agents.judge.JudgeAgent import JudgeAgent
from src.core.LongMemEvalDataset import LongMemEvalDataset
from src.core.Config import Config
from src.models.Model import Model
from src.models.QwenModel import QwenModel
from src.agents.rag.RAGAgent import RAGAgent
from src.agents.full_context.FullContextAgent import FullContextAgent


def load_memory_model(config: Config):
    if config.memory_model.model_type == "transformers":
        return QwenModel(config.memory_model.model_name, quantized=config.memory_model.quantized)
    elif config.memory_model.model_type == "litellm":
        return LiteLLMModel(config.memory_model.model_name)
    else:
        raise ValueError(f"Invalid model type: {config.memory_model.model_type}")


def load_memory_agent(memory_model: Model, config: Config):
    if config.memory_agent == "RAG":
        return RAGAgent(model=memory_model)
    elif config.memory_agent == "FullContext":
        return FullContextAgent(model=memory_model)
    else:
        raise ValueError(f"Invalid memory agent: {config.memory_agent}")


def run_experiment(config: Config):
    memory_model = load_memory_model(config)

    judge_model = LiteLLMModel(config.judge_model_name)
    judge_agent = JudgeAgent(model=judge_model)

    memory_agent = load_memory_agent(memory_model, config)

    correct_predictions = 0
    longmemeval_o_dataset = LongMemEvalDataset(config.longmemeval_dataset_type)

    for instance in longmemeval_o_dataset[: config.N]:
        predicted_answer = memory_agent.answer(instance)
        answer_is_correct = judge_agent.judge(instance, predicted_answer)
        if answer_is_correct:
            correct_predictions += 1

        print(f"Question: {instance.question}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Answer: {instance.answer}")
        print(f"Correct: {answer_is_correct}")
        print("-" * 100)

    print(f"Correct predictions: {correct_predictions}/{config.N}")
    print(f"Accuracy: {correct_predictions/config.N}")
