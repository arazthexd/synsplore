from typing import List
import os

from omegaconf import DictConfig
from hydra.utils import instantiate, get_original_cwd, to_absolute_path

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcNumRotatableBonds

from druglab.storage import MolStorage, RxnStorage
from druglab.featurize import BaseFeaturizer

def initiate_stores(cfg: DictConfig):
    
    # LOAD AND FILTER MOLS
    mols = MolStorage()
    [mols.load_mols(
        os.path.join(get_original_cwd(), mfile) 
    ) for mfile in cfg.run.inputs.molecules.files]

    if cfg.run.checking:
        mols.subset([i for i in range(min(len(mols), 1500))], inplace=True)

    uweight = cfg.run.inputs.molecules.filters.upper_weight
    lweight = cfg.run.inputs.molecules.filters.lower_weight
    maxrot = cfg.run.inputs.molecules.filters.max_rotbonds
    mols.subset([
        i for i, mol in enumerate(mols)
        if CalcExactMolWt(mol) < uweight
        and CalcExactMolWt(mol) > lweight
        and CalcNumRotatableBonds(mol) < maxrot
    ], inplace=True)

    # LOAD RXNS
    rxns = RxnStorage()
    [rxns.load_rxns(
        os.path.join(get_original_cwd(), rxnfile) 
    ) for rxnfile in cfg.run.inputs.reactions.files]

    # REMOVE MOLS NOT MATCHING TO ANY REACTION
    mid2rxnrids = rxns.match_mols(mols)
    mols.subset(list(mid2rxnrids.keys()), inplace=True)

    # # FEATURIZE MOLS
    # mfcfg = cfg.features.molecules
    # if mfcfg.output.reuse_input:
    #     mol_fzer: BaseFeaturizer = instantiate(mfcfg.input.featurizer)
    # else:
    #     mol_fzer: BaseFeaturizer = instantiate(mfcfg.output.featurizer)
    # mol_fzer.fit(mols.objects)
    # mols.featurize(mol_fzer,
    #                overwrite=True,
    #                n_workers=cfg.run.num_workers)

    # ADD FEATURIZED MOLS TO RXNS AND REMOVE UNMATCHED RXNS
    rxns.add_mols(mols, overwrite=True)
    rxns.clean()

    # # FEATURIZE RXNS
    # rxnfcfg = cfg.features.reactions
    # if rxnfcfg.output.reuse_input:
    #     rxn_fzer: BaseFeaturizer = instantiate(rxnfcfg.input.featurizer)
    # else:
    #     rxn_fzer: BaseFeaturizer = instantiate(rxnfcfg.output.featurizer)
    # rxn_fzer.fit(rxns.objects)
    # rxns.featurize(rxn_fzer,
    #                overwrite=True,
    #                n_workers=cfg.run.num_workers)
    
    # SAVE!
    if cfg.run.outputs.use_original_dir:
        out_dir = get_original_cwd()
    else:
        out_dir = os.getcwd()
        
    out_path = os.path.join(
        out_dir,
        cfg.run.outputs.reactions
    )
    dir_path = os.path.dirname(out_path)
    os.makedirs(dir_path, exist_ok=True)
    rxns.save(dst=out_path, save_mols=True)

    out_path = os.path.join(
        out_dir,
        cfg.run.outputs.molecules
    )
    dir_path = os.path.dirname(out_path)
    os.makedirs(dir_path, exist_ok=True)
    mols.save(dst=out_path)

    