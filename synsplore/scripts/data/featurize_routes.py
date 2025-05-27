from typing import List
import os
import h5py
from mpire import WorkerPool

import numpy as np

from omegaconf import DictConfig
from hydra.utils import instantiate, get_original_cwd, to_absolute_path

from rdkit import Chem
from rdkit.Chem import QED

from druglab.storage import MolStorage, RxnStorage
from druglab.synthesis import SynRouteSampler, SynRouteStorage
from druglab.featurize import BaseFeaturizer, CompositeFeaturizer

def _get_featurizer(cfg: DictConfig):
    infzer = instantiate(cfg.input.featurizer)
    if cfg.output.reuse_input:
        return infzer
    outfzer = instantiate(cfg.output.featurizer)
    return CompositeFeaturizer([infzer, outfzer])

def featurize_routes(cfg: DictConfig):

    n_workers = cfg.run.num_workers

    if not cfg.run.outputs.use_original_dir:
        base_dir = os.getcwd()
    else:
        base_dir = get_original_cwd()

    # LOAD ROUTES
    routes = SynRouteStorage.load(os.path.join(
        base_dir, cfg.run.outputs.routes
    ))
    
    # INITIATE FEATURIZERS
    mol_featurizer: BaseFeaturizer = _get_featurizer(
        cfg.features.molecules
    )
    rxn_featurizer: BaseFeaturizer = _get_featurizer(
        cfg.features.reactions
    )

    # FIT FEATURIZERS
    mol_featurizer.fit(routes.rstore.objects + routes.pstore.objects)
    rxn_featurizer.fit(routes.rxnstore.objects)

    # FEATURIZE
    routes.rstore.featurize(mol_featurizer, overwrite=True, n_workers=n_workers)
    routes.pstore.featurize(mol_featurizer, overwrite=True, n_workers=n_workers)
    routes.rxnstore.featurize(rxn_featurizer, overwrite=True, n_workers=n_workers)

    # SAVE
    routes.save(os.path.join(
        base_dir, cfg.run.outputs.routes
    ))

    