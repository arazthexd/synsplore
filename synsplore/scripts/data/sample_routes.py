from typing import List
import os
import h5py
from mpire import WorkerPool
import tqdm
import gc

import numpy as np

from omegaconf import DictConfig
from hydra.utils import instantiate, get_original_cwd, to_absolute_path

from rdkit import Chem
from rdkit.Chem import QED
import medchem as mc
from medchem.functional import nibr_filter

from druglab.storage import MolStorage, RxnStorage
from druglab.synthesis import SynRouteSampler, SynRouteStorage
from druglab.featurize import BaseFeaturizer, CompositeFeaturizer

def sample_routes(cfg: DictConfig):

    n_workers = cfg.run.num_workers

    if not cfg.run.outputs.use_original_dir:
        base_dir = os.getcwd()
    else:
        base_dir = get_original_cwd()

    # LOAD RXNS
    rxns = RxnStorage.load(os.path.join(
        base_dir,
        cfg.sampling.inputs.reactions
    ))

    # INITIATE SAMPLER
    samp_opts = cfg.sampling.options
    sampler = SynRouteSampler(
        min_steps=samp_opts.min_steps,
        max_steps=samp_opts.max_steps,
        n_template_batch=samp_opts.batch_num_templates,
        n_route_batch=samp_opts.batch_num_samplers_per_template
    )

    # INITIATE DATABASE
    if not cfg.run.outputs.use_original_dir:
        db_path = os.path.join(
            base_dir,
            cfg.run.outputs.routes
        )
    else:
        db_path = to_absolute_path(
            cfg.run.outputs.routes
        )
    # os.makedirs(db_path, exist_ok=True)
    db = h5py.File(db_path, "w")
    # db.create_group("seqs")
    # db.create_group("rstore")
    # db.create_group("pstore")
    # db.create_group("rxnstore")
    db.create_group("batches")
    db.close()

    # INITIATE FILTERS
    alerts = mc.catalogs.NamedCatalogs.nibr()

    # SAMPLE UNTIL TARGET IS REACHED
    target = (1000 
              if cfg.run.checking
              else cfg.sampling.output.target_num)
    cn = 0
    batch_id = 0
    while cn < target:
        routes = sampler.sample(
            rxns, 
            only_final=samp_opts.final_routes_only, 
            num_processes=n_workers,
            return_storage=False
        )

        def filter_func(route):
            if not QED.qed(route.products[-1]) > cfg.sampling.filters.min_qed:
                return False
            if cfg.sampling.filters.alerts:
                if alerts.HasMatch(route.products[-1]):
                    return False
            return True

        with WorkerPool(n_workers) as pool:
            mask = pool.map(
                filter_func,
                routes,
                progress_bar=True
            )
        routes = [routes[i] for i, m in enumerate(mask) if m]

        routes = SynRouteStorage(routes)
        
        db = h5py.File(db_path, "a")
        grp = db["batches"].create_group(str(batch_id))
        batch_id += 1
        routes.save(grp, close=False)
        # for i in range(cn, cn+len(routes)):
        #     db[f"seqs/{i}"] = h5py.SoftLink(f"/batches/{batch_id}/seqs/{cn-i}")
        #     db[f"rstore"]

        cn += len(routes)

        print(f"Routes sampled: {len(routes)} | Total: {cn}")
        db.close()
        del routes
        gc.collect()
    
    # CREATE A ROUTE STORAGE AND SAVE
    routes = SynRouteStorage(routes)

    out_path = os.path.join(
        base_dir,
        cfg.run.outputs.routes
    )
    dir_path = os.path.dirname(out_path)
    os.makedirs(dir_path, exist_ok=True)
    routes.save(out_path)

    