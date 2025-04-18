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

def load_yaml_config(config_path):
    """Load YAML configuration from the given file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@click.group()
def cli():
    """
    Synsplore CLI: A command-line interface for synsplore.
    """
    pass

@cli.command()
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

@cli.command()
@click.option("-c", "--config", type=click.Path(), default=None,
              help="Path to the config file. cli options overwrite this.")
@click.option("-wd", "--working-dir", type=click.Path(), default=None,
              help="Working directory for inputs and outputs.")
@click.option("-m", "--molecules", type=click.Path(), default=None,
              help="Prepared building blocks for the sampling. (.pkl)")
@click.option("-r", "--reactions", type=click.Path(), default=None,
              help="Prepared reactions for the sampling. (.pkl)")
@click.option("-i", "--molmap", type=click.Path(), default=None,
              help="Mapping from mol idx to reaction idx. (.pkl)")
@click.option("-o", "--output", type=click.Path(), default=None,
              help="Output file for the sampled synthesis routes. (.pkl)")
@click.option("-ns", "--num-samples", type=int, default=None,
              help="Number of synthesis routes to generate.")
@click.option("--debug", is_flag=True, default=False,
              help="Whether to run the script in debug mode (verbose).")
def sample(config,
           working_dir,
           molecules, 
           reactions, 
           molmap,
           output,
           num_samples, 
           debug):
    """
    For any given number of molecule or reaction files, creates a storage
    for all the molecules and reactions, matches molecules to reactions,
    and saves their respective storage objects.
    """

    default = {
            "working_dir": "./out/",
            "molecules": "mols.pkl",
            "reactions": "rxns.pkl",
            "molmap": "molmap.pkl",
            "output": "routes.pkl",
            "num_samples": 1000
        }

    if not debug:
        rdBase.DisableLog("rdApp.*")

    if config is not None:
        config = load_yaml_config(config)
        default.update(config)

    config = default
    working_dir = working_dir or config["working_dir"]
    molecules = molecules or config["molecules"]
    reactions = reactions or config["reactions"]
    molmap = molmap or config["molmap"]
    output = output or config["output"]
    num_samples = num_samples or config["num_samples"]

    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    molecules = os.path.join(working_dir, molecules)
    reactions = os.path.join(working_dir, reactions)
    molmap = os.path.join(working_dir, molmap)
    output = os.path.join(working_dir, output)

    click.echo("Loading molecules, reactions, and molmap...")

    with open(molecules, "rb") as f:
        mols = dill.load(f)

    with open(reactions, "rb") as f:
        rxns = dill.load(f)

    with open(molmap, "rb") as f:
        midx2rrxnids = dill.load(f)

    opts = SynRouteSamplerOpts(min_steps=3, 
                               max_steps=6, 
                               max_construct_attempts=30,
                               max_sample_attempts=20)
    sampler = SynRouteSampler(bbs=mols, 
                              rxns=rxns,
                              bb2rxnr=midx2rrxnids,
                              processed=True,
                              options=opts)
    
    def filter_func(route: SynthesisRoute):
        return rdMolDescriptors.CalcNumRotatableBonds(route.products[-1]) < 8

    def worker_init(worker_state):
        worker_state["filter_func"] = filter_func
        worker_state["sampler"] = sampler

    def worker_task(worker_state, idx):
        sampler: SynRouteSampler = worker_state["sampler"]
        filter_func = worker_state["filter_func"]
        return sampler.sample(only_last=False, 
                              filter_func=filter_func)

    click.echo("Sampling...")
    synroutes: List[SynthesisRoute] = []
    bar = tqdm(total=num_samples)
    while len(synroutes) < num_samples:

        with WorkerPool(n_jobs=8, use_worker_state=True) as pool:
            o = pool.map(worker_task, 
                         range(100), 
                         worker_init=worker_init)
            routes = [r for rs in o for r in rs]
        
        synroutes.extend(routes)
        bar.update(len(routes))
    
    bar.close()

    click.echo("Saving...")
    with open(output, "wb") as f:
        dill.dump(synroutes, f)

if __name__ == '__main__':
    cli()
