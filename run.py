import os, argparse, statistics
import tqdm

from model.schnet import SchNet
from utils import readData, readLabels, createDataList

import torch
from torch_geometric.loader import DataLoader

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train(args: argparse.Namespace, loader: DataLoader): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SchNet(
            hidden_channels=args.hidden_channels,
            num_filters=args.num_filters,
            num_interactions=args.num_interactions,
            num_gaussians=args.num_gaussians,
            cutoff=args.cutoff,
            max_num_neighbors=args.max_neighbors,
            readout=args.readout,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    critereon = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        maes = []
        for data in tqdm(loader):
            optimizer.zero_grad()
            data = data.to(device)

            prediction = model(data.emb, data.pos, data.idx)
            loss = critereon(prediction, data.y)
            loss.backward()

            optimizer.step()

            maes.append(loss)

        if epoch % args.log_interval == 0: 
            print(f'Epoch: {epoch} | MAE: {statistics.mean(maes):.5f} ± {statistics.stdev(maes):.5f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels','-h', type=int, default=128)
    parser.add_argument('--num_filters', '-f', type=int, default=128)
    parser.add_argument('--num_interactions', '-i', type=int, default=6)
    parser.add_argument('--cutoff', '-c', type=float, default=20.0)
    parser.add_argument('--max_neighbors', '-mn', type=int, default=32)
    parser.add_argument('--readout', '-r', type=str, default='add')
    parser.add_argument('--batch_size', '-b', type=int, default=10)
    parser.add_argument('--epochs', '-e', type=int, default=20)
    parser.add_argument('--log_interval', '-li', type=int, default=5)
    parser.add_argument('--esm_embed_path', '-emb', type=str)
    parser.add_argument('--pdb_path', '-p', type=str)
    parser.add_argument('--labels_file', '-l', type=str)

    args = parser.parse_args()

    positions, embeddings = readData(args.pdb_path, args.esm_embed_path)
    labels = readLabels(args.labels_file)
    
    data = createDataList(positions, embeddings, labels)
    loader = DataLoader(data, args.batch_size)

    train(args, loader)



