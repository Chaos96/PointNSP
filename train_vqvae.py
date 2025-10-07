import torch
import torch.optim as optim
import torch.nn as nn
from model.vqvae_model import VQVAE
from datasets.data_processing import get_dataset

def main():
    # Hyperparameters
    input_dim = 3  # x, y, z coordinates
    hidden_dim = 1024
    codebook_size = 5192
    embedding_dim = 1024
    batch_size = 5
    num_epochs = 200
    learning_rate = 5e-3
    num_points = 2048
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    # Initialize model
    model = VQVAE(input_dim, hidden_dim, codebook_size, embedding_dim, num_points).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load ShapeNetV2 data
    root_dir = "data/ShapeNetCore.v2.PC15k"
    categories = 'car'  # Example: cars and chairs
    train_dataset, test_dataset = get_dataset(root_dir, num_points, category=categories)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Train the model
    for epoch in range(num_epochs):
        total_loss = 0
        total_chamfer_loss = 0
        total_vq_loss = 0
        
        for batch in dataloader:
            x = batch['train_points'].to(device)
            optimizer.zero_grad()
            
            reconstructed, loss, chamfer_loss, vq_loss = model(x)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_chamfer_loss += chamfer_loss.item()
            total_vq_loss += vq_loss.item()

            print(f"Loss: {total_loss:.4f}, ")
    
        avg_loss = total_loss / len(dataloader)
        avg_chamfer_loss = total_chamfer_loss / len(dataloader)
        avg_vq_loss = total_vq_loss / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Loss: {avg_loss:.4f}, "
            f"Chamfer Loss: {avg_chamfer_loss:.4f}, "
            f"VQ Loss: {avg_vq_loss:.4f}")

if __name__ == "__main__":
    main()
