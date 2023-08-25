# SchNetESM

This model modifies the [SchNet model](https://arxiv.org/pdf/1706.08566.pdf) for learning quantum interactions and incorporates ESM embeddings for residues. Here, we train the SchNet model to predict on proteins, rather than individual small molecules.

## Inputs
1. PDB files for each protein of interest
2. ESM embeddings for each protein of interest. For the example data provided, the ESM embeddings were [generated](https://github.com/facebookresearch/esm/tree/main) via the individual chain sequences.
3. The labels for each protein of interest. For a SchNet-based prediction, these should correspond to potential energies.

## Modifications
This model makes several underlying modifications and assumptions to the original SchNet model: 
1. Given the likelihood of few samples for large proteins, we do not utilize every atom in the protein for prediction. Rather, we focus on the C-alpha atoms as representations for the corresponding residues, as this should provide a good correlation to understanding residue-level potential energy and prevent overfitting.
2. Because of (1), and because we are utilizing ESM embeddings, we do not utilize nuclear charge information for computing potentials. It is probable that most energetic interactions between proteins can be learned at the residue level.

### Dependencies
This model depends on several packages: 
1. BioPython
2. PyTorch
3. PyTorch Geometric


### Running/Training
The `run.py` is a script provided to run and train this model. The command is as follows: 

```
usage: python run.py [-h]
                  [--hidden_channels NUMBER_OF_HIDDEN_CHANNELS]
                  [--num_filters  NUM_FILTERS]
                  [--cutoff INTERACTION_DISTANCE_CUTOFF]
                  [--num_interactions NUMBER_OF_INTERACTION_BLOCKS]
                  [--max_neighbors MAXIMUM_NUMBER_OF_NEIGBORS]
                  [--readout AGGREGATION_READOUT]
                  [--batch_size]
                  [--epochs]
                  [--log_interval]
                  [--esm_embed_path]
                  [--pdb_path]
                  [--labels_file]

required arguments:
  esm_embed_path        Path to the directory of ESM embeddings
  pdb_path              Path to the directory of PDB files
  labels_file           Path to the file containing labels
```
