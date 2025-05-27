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
from druglab.pharm import (
    PharmGenerator, BASE_DEFINITIONS_PATH,
    PharmDefaultProfiler, InternalStericAdjuster, PharmProfiler, PharmProfile
)
from druglab.prepare import MoleculePreparation

def profile_products(cfg: DictConfig):
    
    if not cfg.run.outputs.use_original_dir:
        base_dir = os.getcwd()
    else:
        base_dir = get_original_cwd()

    # LOAD PRODUCTS
    routes_path = os.path.join(base_dir, cfg.run.outputs.routes)
    with h5py.File(routes_path, "r") as db:
        pstore = MolStorage.load(db["pstore"])
        products = [pstore[db["seqs"][str(i)][-2, 1]]
                    for i in range(db["seqs"].__len__())]
        
    # PREPARE
    if cfg.run.checking:
        products = products[:10]
    products = MolStorage(products)
    prep: MoleculePreparation = instantiate(cfg.pharms.preparation)
    products.prepare(prep, inplace=True, n_workers=cfg.run.num_workers)
    products = products.subset([i for i, prod in enumerate(products.objects)
                                if prod is not None])

    writer = Chem.SDWriter("prods_test.sdf")
    [writer.write(prod) for prod in products.objects[:10]]
    writer.close()

    # INITIATE PHARM GENERATOR
    if cfg.pharms.definitions == "default":
        def_path = BASE_DEFINITIONS_PATH
    else:
        def_path = cfg.pharms.definitions
    pgen = PharmGenerator()
    pgen.load_file(def_path)

    # INITIATE PHARM PROFILER
    profiler: PharmProfiler = instantiate(cfg.pharms.profiler)(ftypes=pgen.ftypes)

    # PROFILE PRODUCTS
    def task(mol: Chem.Mol):
        profs = []
        for conf in mol.GetConformers():
            idx = conf.GetId()
            pharm = pgen.generate(mol, idx)
            prof = profiler.profile(pharm)
            profs.append(prof)
            # print(
            #     "confid", idx,
            #     "\nsmiles", Chem.MolToSmiles(mol),
            #     "\npharm nfeats", pharm.n_feats,
            #     "\nnames", pharm.ftype_names,
            #     "\nprof subids", prof.subids,
            #     "\npgen types", pgen.ftype_names,
            #     "\ndef_path", def_path,
            #     "\nprofiler types", profiler.ftypes
            # )
        profs = sum(profs)
        return profs
    
    nw = 1 if cfg.run.checking else cfg.run.num_workers
    with WorkerPool(nw) as pool: # TODO 
        profs: list[PharmProfile] = pool.map(task, 
                                             products.objects, 
                                             progress_bar=True)
    
    # SAVE
    with h5py.File(cfg.run.outputs.pharms, "w") as db:
        for i, prof in enumerate(profs):
            grp = db.create_group(str(i))
            grp["tys"] = prof.tys
            grp["tyids"] = prof.tyids
            grp["dists"] = prof.dists
            grp["angles"] = np.arccos(prof.cos)
            subgrp = grp.create_group("subids")
            for j, sids in enumerate(prof.subids):
                subgrp.create_dataset(str(j), data=sids)