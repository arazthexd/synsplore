import click
from tqdm import tqdm
import os, yaml, dill, glob
from mpire.pool import WorkerPool
from typing import List

from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from druglab.storage import MolStorage, RxnStorage
from druglab.synthesis import (
    SynRouteSampler, SynRouteSamplerOpts,
    SynthesisRoute, 
)

from .utils import load_yaml_config

@click.command()
@click.option("-c", "--config", type=click.Path(), default=None,
              help="Path to the config file. cli options overwrite this.")
@click.option("-m", "--molecules", multiple=True, 
              type=click.Path(), default=None,
              help="Molecules to prepare.")
@click.option("-r", "--reactions", multiple=True, 
              type=click.Path(), default=None,
              help="Reactions to prepare.")
@click.option("-od", "--output-dir", 
              type=click.Path(file_okay=False), 
              default="./out/",
              help="Output directory for saving mols and rxns.")
@click.option("-om", "--output-mols", 
              type=click.Path(), default="mols.pkl",
              help="Output file to write the mol storage object to. (.pkl)")
@click.option("-or", "--output-rxns", 
              type=click.Path(), default="rxns.pkl",
              help="Output file to write the rxn storage object to. (.pkl)")
@click.option("-oi", "--output-molmap",
              type=click.Path(), default="molmap.pkl",
              help="Output file to write mappings from mol idx to reactions.")
@click.option("-umw", "--upper-molweight", type=float, default=250,
              help="Upper molecule weight limit.")
def prepare(config,
            molecules, 
            reactions, 
            output_dir, 
            output_mols, 
            output_rxns,
            output_molmap,
            upper_molweight):
    """
    For any given number of molecule or reaction files, creates a storage
    for all the molecules and reactions, matches molecules to reactions,
    and saves their respective storage objects.
    """

    if config is not None:
        config = load_yaml_config(config)
        molecules = molecules or config["molecules"]
        reactions = reactions or config["reactions"]
        output_dir = output_dir or config["output_dir"]
        output_mols = output_mols or config["output_mols"]
        output_rxns = output_rxns or config["output_rxns"]
        output_molmap = output_molmap or config["output_molmap"]
        upper_molweight = upper_molweight or config["upper_molweight"]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    click.echo("Reading and cleaning molecule files...")
    mols = MolStorage()
    for mols_file in tqdm(molecules):
        for f in glob.glob(mols_file):
            mols.load_mols(f)
    mols.clean()
    mols.subset([
        i for i, mol in enumerate(mols) 
        if rdMolDescriptors.CalcExactMolWt(mol) < upper_molweight
    ], inplace=True)
    
    click.echo("Reading reaction files...")
    rxns = RxnStorage()
    for rxns_file in tqdm(reactions):
        for f in glob.glob(rxns_file):
            rxns.load_rxns(f)

    click.echo("Adding matching molecules to reaction storage...")
    rxns.add_mols(mols)
    
    click.echo("Cleaning reactions storage...")
    rxns.clean()
    rxns.subset([
        i for i, rxn in enumerate(rxns)
        if min(len(mstore) for mstore in rxns.mstores[i]) > 10
    ], inplace=True)

    click.echo("Matching final mols to final rxns...")
    midx2rrxnids = rxns.match_mols(mols)
    mols.subset(list(midx2rrxnids.keys()), inplace=True)

    click.echo("Saving...")
    
    with open(os.path.join(output_dir, output_mols), "wb") as f:
        dill.dump(mols, f)
    
    with open(os.path.join(output_dir, output_rxns), "wb") as f:
        dill.dump(rxns, f)

    with open(os.path.join(output_dir, output_molmap), "wb") as f:
        dill.dump(midx2rrxnids, f)