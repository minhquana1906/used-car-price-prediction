import hydra
from evaluate import evaluate_pipeline
from preprocess import preprocess_pipeline
from train import train_pipline


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    preprocess_pipeline(cfg)
    train_pipline(cfg)
    evaluate_pipeline(cfg)


if __name__ == "__main__":
    main()
