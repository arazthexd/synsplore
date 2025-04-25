# Synsplore
Learning to explore the synthesizable chemical space, conditioned on target pharmacophores.

## TODOs
### DrugLab Features
- [ ] Implement PharmStorage class and update the existing code.
- [x] Figure out why base definitions yaml for pharms is not transfered

### Data Generation
- [x] Reading data and creating a storage for mols, rxns, matching
- [ ] Preprocessing filters for dataset molecules
- [x] Featurizing mols, rxns
- [x] Sampling synthesis routes 
- [x] Filter for sampling routes (no duplicates)
- [x] Featurizing synthesis routes 
- [x] Pharm generation for syn route products
- [ ] (?) Dimensionality reduction for molecules
- [ ] Seed for sampling routes

### Dataset
- [x] Main torch dataset (initiating, loading, saving, indexing)
- [x] Pharm feature location/value noise transform
- [x] Pharm feature masking transform
- [ ] Dataset Initiation Script

### Model
TODO...

### Repository
- [ ] Create dev branch...
- [ ] Pytorch Dataset Initiation Description
- [ ] Add torch requirements to dependencies and installation instruction.

## Installation
1. Create a virtual environment either in conda, mamba, or pip.
2. (conda/mamba) Install packages that you want to be installed only from conda/mamba
3. Install synsplore package in editable mode.
```terminal
cd PATH/TO/THIS/REPO
pip install -e .
```

## Usage
### Data Preparing/Preprocessing
For this step, at least one reaction file (a simple text file containing reaction smarts in each line) and one building blocks file (a text file with molecule smiles in each line or an sdf file) is needed. The paths of these files should go below the "molecules" and "reactions" sections of a config file. Our used config file (hopefully with descriptions of each option written when you are reading it) is available here: [prepare.yaml](configs/prepare.yaml)

Next, you will need to run the following command in the shell:
```terminal
synsplore prepare -c configs/prepare.yaml
```
And the outputs will be available by default in the "out" folder. However, this can be changed in the config file as well.

### Synthesis Route Sampling
Next, given the prepared reactions and reactants, synthesis routes/trees will be randomly sampled. For this, the "mols", "rxns", and "molmap" outputs of the preparation step will be needed as inputs to the config file ([sample.yaml](configs/sample.yaml)). The below command will then need to be run, given the config file is at the default location:
```terminal
synsplore sample -c configs/sample.yaml
```

### Molecules, Reactions, and Routes Featurization
The "mols" and "rxns" outputs from the first step, along with the "routes" output from the second step will need to be featurized. The featurized routes are used for training the model. These features include:
1. An updated "routes" object, with elements of it (reactants, reactions, etc.) featurized inside it.
2. A "pharms" object which includes featurized pharmacophores of the final products of every route in "routes".

```terminal
synsplore featurize routes -c configs/roufeats.yaml 
```

The featurized "mols" and "rxns" objects will not be used for training purposes. These will be used for inference, where a nearest neighbor model will be used on them to search for closest available items to the model's suggestions.

```terminal
synsplore featurize molecules -c configs/molfeats.yaml
synsplore featurize reactions -c configs/rxnfeats.yaml
```

### PyTorch Dataset Initiation
TODO...

### Model Training
TODO...

### Model Inference
TODO...