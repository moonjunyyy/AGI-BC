import torch

x = torch.nn.Linear(3, 4)
optim = torch.optim.Adam(x.parameters(), lr=0.001)
input_T = torch.randn(2, 3)
target_T = torch.randn(2, 4)
for i in range(100):
    optim.zero_grad()
    output_T = x(input_T)
    loss = torch.nn.functional.mse_loss(output_T, target_T)
    loss.backward()
    # optim.step()
    print(loss)
    print(x.weight)