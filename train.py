import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchinfo import summary


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    dataset = instantiate(cfg.dataset)
    model = instantiate(cfg.model)

    print(dataset)
    summary(model)


if __name__ == "__main__":
    main()
