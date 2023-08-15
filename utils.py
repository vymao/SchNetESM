import os 
import collections
from typing import Tuple

from Bio.PDB import PDBParser
from Bio import SeqIO

import torch
import torch_geometric
from torch_geometric.data import Data

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def createDataList(positions: dict, embeddings: dict, labels: dict) -> list: 
    """
    This function creates the list of Data objects for training.

    Parameters: 
        positions
        embeddings
        labels

    Returns: 
        data_list - A list of Data objects, one for each protein.
    """
    data_list = []
    for idx, protein in enumerate(positions.keys()): 
        pos, emb, y = positions[protein], embeddings[protein], labels[protein]
        data_list.append(Data(pos=pos, emb=emb, y=y, id=idx))

    return data_list

def readLabels(path: str) -> dict:
    """
    This function reads in the label file and returns the labels.

    Parameters: 
        path - The path to the label file

    Returns: 
        labels - A dictionary containing the labels for each protein
    """ 
    labels = {}
    with open(path, 'r') as file: 
        for line in file: 
            protein, label = line.rstrip().split(' ')
            labels[protein] = label

    return labels

def readData(pdb_path: str, embed_path: str) -> Tuple[dict]: 
    """
    This function reads in the PDB files and returns the corresponding sequence lengths for each chain. 

    Parameters: 
        pdb_path - The path to the directory holding the PDB files
        embed_path - The path to the embedding files
    
    Returns: 
        (positions, embeddings) - A tuple containing the positions of the C-alpha atoms and the residue 
            embeddings for each protein.
    """
    files = os.listdir(pdb_path)

    positions = {}
    embeddings = {}

    parser = PDBParser(PERMISSIVE = True, QUIET = True)

    for file in files: 
        filepath = os.path.join(pdb_path, file)
        protein = os.path.basename(file).split('.')[0]
        structure = parser.get_structure(protein, filepath)
        model = list(structure.get_models())[0]

        for chain in model.get_chains(): 
            # Get the C-alpha positions for this chain.
            for residue in chain.get_residues(): 
                c_alpha = torch.Tensor([list(residue['CA'].get_vector())])
                if protein in positions: 
                    positions[protein] = torch.concat([positions[protein], c_alpha], dim = 0)
                else: 
                    positions[protein] = c_alpha


            prot_chain_id = protein + ':' + chain.id 
            # Get the sequence length for this chain.
            sequence_length = len(list(chain.get_residues()))

            # Get the embeddings for this chain.
            seq_embeddings = getEmbeddings(protein ,chain, embed_path)
            seq_embeddings = seq_embeddings[:sequence_length]
            if protein in embeddings: 
                embeddings[protein] = torch.concat([embeddings[protein], seq_embeddings], dim = 0)
            else: 
                embeddings[protein] = seq_embeddings
    
    return (positions, embeddings)

def getEmbeddings(protein: str, chain: str, path: str) -> torch.Tensor: 
    """
    This function reads in the ESM embedding files and returns the loaded embeddings for each residue on the corresponding
    protein and chain.

    Parameters: 
        protein - The protein name
        chain - The chain id
        path - The path to the embedding files

    Returns: 
        chain_embed - The residue-level embeddings
    """
        
    file_name = protein + ':' + chain.id  
    embeddings_path = os.path.join(path, file_name + '.pt')
    
    # We get the 33rd layer representation of the embeddings, per the examples on the ESM repo.
    chain_embed = torch.load(embeddings_path)['representations'][33]
    return chain_embed



