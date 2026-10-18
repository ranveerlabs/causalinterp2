import torch

iterations = 100

w=torch.randn(2, 3, device='cuda')
x=torch.tensor([[1.0, 2.0]], device='cuda')
trg = torch.tensor([[1.0, 0.0, 1.0]], device='cuda')
rateL= 0.01

for iteration in range(iterations):
    y=x@w
    res=y-trg
    gradY = 2*res/3
    gradW= x.T@gradY
    loss= (res**2).mean()
    w = w - rateL*gradW
    if iteration % 10 == 0:
        print('iteration =',iteration,', loss =', loss)

