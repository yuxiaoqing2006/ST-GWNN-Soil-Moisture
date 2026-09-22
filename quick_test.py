"""
Quick‑test script for ST‑GWNN (CAGEO journal requirement)
No real soil‑moisture dataset required.
Generate synthetic dummy data to verify model executable.
Run: python quick_test.py
"""
import torch
import torch.nn as nn


from models.st_gwnn_optimized import ST_GWNN_Optimized

# ---------------------- Config (match paper setting) ----------------------
batch_size = 2
num_nodes = 6
seq_len = 30
input_dim = 7
output_dim = 1

model_config = {
    "num_nodes": num_nodes,
    "input_dim": input_dim,
    "hidden_dim": 136,
    "output_dim": output_dim,
    "seq_len": seq_len,
    "num_gwnn_layers": 3,
    "num_temporal_blocks": 3,
    "num_wavelet_scales": 3,
    "dropout": 0.1
}

# ---------------------- Create synthetic dummy data ----------------------
# Input shape: (batch, num_nodes, seq_len, input_dim)
x_dummy = torch.randn(batch_size, num_nodes, seq_len, input_dim)
y_dummy = torch.randn(batch_size, num_nodes, output_dim)

# ---------------------- Init model ----------------------
model = ST_GWNN_Optimized(model_config)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e‑3)

print("="*60)
print("ST‑GWNN Quick‑Test start (synthetic dummy data)")
print(f"Model total parameters: {sum(p.numel() for p in model.parameters()):,}")
print("="*60)

# ---------------------- Short training loop ----------------------
for epoch in range(3):
    optimizer.zero_grad()
    pred = model(x_dummy) 
    loss = criterion(pred, y_dummy)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1:2d} | Loss = {loss.item():.4f}")

print("\n✅ Quick‑test PASSED! Model forward & backward work normally.")
print("Note: This test uses synthetic dummy data, NOT real observation dataset.")
