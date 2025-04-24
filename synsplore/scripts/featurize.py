import click, dill, os
from tqdm import tqdm
from mpire.pool import WorkerPool

from rdkit import Chem
from rdkit.Chem import rdDistGeom

from druglab.featurize import (
    NAME2FEATURIZER, BaseFeaturizer,
    MorganFPFeaturizer, 
    RxnOneHotFeaturizer
)
from druglab.storage import MolStorage, RxnStorage
from druglab.synthesis import (
    SynRouteStorage, SynRouteFeaturizer, SynthesisRoute
)
from druglab.pharm import PharmGenerator, BASEDEF_PATH

from .utils import load_yaml_config

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

@featurize.command()
@click.option("-c", "--config", type=click.Path(), default=None,
              help="Path to the config file. cli options overwrite this.")
@click.option("-wd", "--working-dir", type=click.Path(), default=None,
              help="Working directory for inputs and outputs.")
@click.option("-r", "--routes", type=click.Path(), default=None,
              help="Routes to featurize. (.pkl)")
@click.option("-mf", "--mol-featurizer", type=click.Path(), default=None,
              help="Featurizer to use for molecules. (.pkl)")
@click.option("-rf", "--rxn-featurizer", type=click.Path(), default=None,
              help="Featurizer to use for reactions. (.pkl)")
@click.option("-or", "--output-routes", type=click.Path(), default=None,
              help="Output file to write the route storage object to. (.pkl)")
@click.option("-op", "--output-pharms", type=click.Path(), default=None,
              help="Output file to write the pharm storage object to. (.pkl)")
def routes(config,
           working_dir,
           routes,
           mol_featurizer,
           rxn_featurizer,
           output_routes,
           output_pharms):
    """
    Featurize routes in a storage object.
    """

    default = {
        "working_dir": "./out/",
        "routes": "routes.pkl",
        "mol_featurizer": "mfeaturizer.pkl",
        "rxn_featurizer": "rfeaturizer.pkl",
        "output_routes": "routes.pkl",
        "output_pharms": "pharms.pkl"
    }

    click.echo("Loading...")
    if config is not None:
        config = load_yaml_config(config)
        default.update(config)
    
    working_dir = working_dir or default["working_dir"]
    routes = routes or default["routes"]
    mol_featurizer = mol_featurizer or default["mol_featurizer"]
    rxn_featurizer = rxn_featurizer or default["rxn_featurizer"]
    output_routes = output_routes or default["output_routes"]
    output_pharms = output_pharms or default["output_pharms"]
    
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    routes = os.path.join(working_dir, routes)
    mol_featurizer = os.path.join(working_dir, mol_featurizer)
    rxn_featurizer = os.path.join(working_dir, rxn_featurizer)
    output_routes = os.path.join(working_dir, output_routes)
    output_pharms = os.path.join(working_dir, output_pharms)
    
    with open(routes, "rb") as f:
        routes: SynRouteStorage = dill.load(f)
    mol_featurizer = BaseFeaturizer.load(mol_featurizer)
    rxn_featurizer = BaseFeaturizer.load(rxn_featurizer)
    
    click.echo("Featurizing Reactants, Reactions, Products...")
    featurizer = SynRouteFeaturizer(rfeaturizer=mol_featurizer, 
                                    pfeaturizer=mol_featurizer, 
                                    rxnfeaturizer=rxn_featurizer)
    routes.featurize(featurizer, overwrite=True, n_workers=-1)

    click.echo("Generating pharmacophores for final products...")
    
    def task(route: SynthesisRoute):
        mol = route.products[-1]
        mol = Chem.AddHs(mol)
        rdDistGeom.EmbedMultipleConfs(mol, numConfs=10, maxAttempts=100)
        phs = [generator.generate(mol, i)
               for i in range(mol.GetNumConformers())]
        mol = Chem.RemoveHs(mol)
        if len(phs) == 0:
            return [], mol
        [ph.infere_distances() for ph in phs]
        return phs, mol
    
    pharms = [] # TODO: Move to Pharmacophore Storage....
    generator = PharmGenerator()
    generator.read_yaml(BASEDEF_PATH)
    with WorkerPool() as pool:
        out = pool.map(task, routes.objects, progress_bar=True)
    remove_idx = [i for i, (phs, mol) in enumerate(out) if len(phs) == 0]
    routes.subset([i for i, route in enumerate(routes.objects) 
                   if i not in remove_idx], inplace=True)
    pharms, prods = zip(*out)
    pharms = [phs for i, phs in enumerate(pharms) if i not in remove_idx]
    prods = [mol for i, mol in enumerate(prods) if i not in remove_idx]

    routes: SynRouteStorage
    for i, route in enumerate(routes):
        routes.pstore.objects[route.seq[-2].idx] = prods[i]
    routes.featurizers = [None]
    routes.rstore.featurizers = [None]
    routes.pstore.featurizers = [None]
    routes.rxnstore.featurizers = [None]

    click.echo("Saving...")
    with open(output_routes, "wb") as f:
        dill.dump(routes, f)

    with open(output_pharms, "wb") as f:
        dill.dump(pharms, f)

    



