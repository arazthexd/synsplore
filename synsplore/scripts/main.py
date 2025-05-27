from omegaconf import DictConfig, OmegaConf
import hydra
import os

from synsplore.scripts.data import (
    initiate_stores, sample_routes, featurize_routes, profile_products
)

@hydra.main(version_base=None, 
            config_path="../../configs", 
            config_name="main")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    
    if cfg.run == "datagen":
        if cfg.subrun == "initiate_stores":
            initiate_stores(cfg["data"])
        elif cfg.subrun == "sample_routes":
            sample_routes(cfg["data"])
        elif cfg.subrun == "featurize_routes":
            featurize_routes(cfg["data"])
        elif cfg.subrun == "all":
            initiate_stores(cfg["data"])
            sample_routes(cfg["data"])
            featurize_routes(cfg["data"])
            profile_products(cfg["data"])
        else:
            raise NotImplementedError()
    
    else:
        raise NotImplementedError()
    
if __name__ == "__main__":
    main()
