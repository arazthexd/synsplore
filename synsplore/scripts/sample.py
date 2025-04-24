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
    SynthesisRoute, SynRouteStorage
)

from .utils import load_yaml_config

@click.command()
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

    opts = SynRouteSamplerOpts(min_steps=1, 
                               max_steps=6, 
                               max_construct_attempts=50,
                               max_sample_attempts=10)
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
        routes = sampler.sample(only_last=False, 
                                filter_func=filter_func)
        routes_ = []
        prevsmis = set()
        for route in routes:
            smi = Chem.MolToSmiles(route.products[-1])
            if smi not in prevsmis:
                routes_.append(route)
                prevsmis.add(smi)
        
        return routes_

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
    synroutes = SynRouteStorage(synroutes)
    with open(output, "wb") as f:
        dill.dump(synroutes, f)