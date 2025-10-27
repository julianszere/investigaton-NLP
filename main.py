from dotenv import load_dotenv
from src.experiments.pass_whole_context.longmemeval_experiment import run_experiment


load_dotenv()

if __name__ == "__main__":
    run_experiment(N=10)
