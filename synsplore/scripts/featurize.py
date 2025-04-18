import click, dill, os

from .utils import load_yaml_config

from druglab.featurize import (
    NAME2FEATURIZER, BaseFeaturizer,
    MorganFPFeaturizer, 
    RxnOneHotFeaturizer
)
from druglab.storage import MolStorage, RxnStorage

@click.group()
def featurize():
    """
    Commands for featurizing various objects.
    """

@featurize.command()
@click.option("-c", "--config", type=click.Path(), default=None,
              help="Path to the config file. cli options overwrite this.")
@click.option("-wd", "--working-dir", type=click.Path(), default=None,
              help="Working directory for inputs and outputs.")
@click.option("-m", "--molecules", type=click.Path(), default=None,
              help="Molecules to featurize. (.pkl)")
@click.option("-f", "--featurizer", type=str, default=None,
              help="Featurizer to use. (default: morgan3-1024)")
@click.option("-om", "--output-mols", type=click.Path(), default=None,
              help="Output file to write the mol storage object to. (.pkl)")
@click.option("-of", "--output-featurizer", type=click.Path(), default=None,
              help="Output file to write the featurizer object to. (.pkl)")
def molecules(config,
              working_dir,
              molecules,
              featurizer,
              output_mols,
              output_featurizer):
    """
    Featurize molecules in a storage object.
    """

    default = {
        "working_dir": "./out/",
        "molecules": "mols.pkl",
        "featurizer": "morgan3-1024",
        "output_mols": "mols.pkl",
        "output_featurizer": "mfeaturizer.pkl"
    }

    click.echo("Loading...")
    if config is not None:
        config = load_yaml_config(config)
        default.update(config)
    
    working_dir = working_dir or default["working_dir"]
    molecules = molecules or default["molecules"]
    featurizer = featurizer or default["featurizer"]
    output_mols = output_mols or default["output_mols"]
    output_featurizer = output_featurizer or default["output_featurizer"]
    featurizer_name = featurizer
    featurizer = NAME2FEATURIZER.get(featurizer_name, 
                                     MorganFPFeaturizer(radius=3, size=1024))
    featurizer: BaseFeaturizer
    
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    molecules = os.path.join(working_dir, molecules)
    output_mols = os.path.join(working_dir, output_mols)
    output_featurizer = os.path.join(working_dir, output_featurizer)
    
    with open(molecules, "rb") as f:
        mols: MolStorage = dill.load(f)

    featurizer.fit(mols.objects)
    
    click.echo("Featurizing...")
    mols.featurize(featurizer=featurizer,
                   overwrite=True,
                   n_workers=-1)
    mols.featurizers = [featurizer_name]
    
    click.echo("Saving...")
    with open(output_mols, "wb") as f:
        dill.dump(mols, f)
    
    featurizer.save(output_featurizer)

@featurize.command()
@click.option("-c", "--config", type=click.Path(), default=None,
              help="Path to the config file. cli options overwrite this.")
@click.option("-wd", "--working-dir", type=click.Path(), default=None,
              help="Working directory for inputs and outputs.")
@click.option("-r", "--reactions", type=click.Path(), default=None,
              help="Reactions to featurize. (.pkl)")
@click.option("-f", "--featurizer", type=str, default=None,
              help="Featurizer to use. (default: rxn-onehot)")
@click.option("-or", "--output-rxns", type=click.Path(), default=None,
              help="Output file to write the rxn storage object to. (.pkl)")
@click.option("-of", "--output-featurizer", type=click.Path(), default=None,
              help="Output file to write the featurizer object to. (.pkl)")
def reactions(config,
              working_dir,
              reactions,
              featurizer,
              output_rxns,
              output_featurizer):
    """
    Featurize reactions in a storage object.
    """

    default = {
        "working_dir": "./out/",
        "reactions": "rxns.pkl",
        "featurizer": "rxn-onehot",
        "output_rxns": "rxns.pkl",
        "output_featurizer": "rfeaturizer.pkl"
    }

    click.echo("Loading...")
    if config is not None:
        config = load_yaml_config(config)
        default.update(config)
    
    working_dir = working_dir or default["working_dir"]
    reactions = reactions or default["reactions"]
    featurizer = featurizer or default["featurizer"]
    output_rxns = output_rxns or default["output_rxns"]
    output_featurizer = output_featurizer or default["output_featurizer"]
    featurizer_name = featurizer
    featurizer = NAME2FEATURIZER.get(featurizer_name,
                                     RxnOneHotFeaturizer())
    featurizer: BaseFeaturizer

    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    reactions = os.path.join(working_dir, reactions)
    output_rxns = os.path.join(working_dir, output_rxns)
    output_featurizer = os.path.join(working_dir, output_featurizer)
    
    with open(reactions, "rb") as f:
        rxns: RxnStorage = dill.load(f)

    featurizer.fit(rxns.objects)
    
    click.echo("Featurizing...")
    rxns.featurize(featurizer=featurizer,
                   overwrite=True,
                   n_workers=-1)
    rxns.featurizers = [featurizer_name]
    
    click.echo("Saving...")
    with open(output_rxns, "wb") as f:
        dill.dump(rxns, f)
    
    featurizer.save(output_featurizer)
