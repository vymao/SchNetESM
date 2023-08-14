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

