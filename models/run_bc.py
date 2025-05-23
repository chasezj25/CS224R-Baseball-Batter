import argparse
from agents import BCAgent
from datasets import SwingDataset
from torch.utils.data import DataLoader, random_split
import torch

def train(agent, dataloader, device):
    agent.model.train()
    total_loss = 0
    for batch in dataloader:
        loss = agent.update(batch, device)
        total_loss += loss
    return total_loss / len(dataloader)

def evaluate(agent, dataloader, device):
    agent.model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            loss = agent.evaluate(batch, device)
            total_loss += loss
    return total_loss / len(dataloader)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SwingDataset(args.data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    agent = BCAgent().to(device)

    for epoch in range(args.epochs):
        train_loss = train(agent, train_loader, device)
        val_loss = evaluate(agent, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Final evaluation
    test_loss = evaluate(agent, val_loader, device)
    print(f"Final Evaluation Loss: {test_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to swing dataset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    main(args)