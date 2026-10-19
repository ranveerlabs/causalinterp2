import torch

iterations = 250
w1=torch.randn(2,4, device='cuda')
w2=torch.randn(4,1, device='cuda')
x=torch.tensor([
    [1.0,2.0],
    [2.0,3.0],
    [3.0,3.0],
    [2.0,6.0]
],device='cuda')
y=h@w2
trg = torch.tensor([
    [3],
    [5],
    [7],
    [9]
],device='cuda')
rateL= 0.0023
for iteration in range(iterations):
    z=x@w1
    h=z*(z>0)
    y=h@w2
    assert y.shape == trg.shape
    res=y-trg
    gradY=2*res/res.numel()
    gradW2= h.T@gradY
    gradH=gradY@w2.T
    gradZ=gradH*(z>0)
    gradW1=x.T@gradZ
    w1=w1-rateL*gradW1
    w2=w2-rateL*gradW2
    loss=(res**2).mean()
    if iteration % 25 == 0:
        print('iteration =',iteration,', loss =', loss)
test=torch.tensor([[2.0,3.0]], device='cuda')
tz=test@w1
th=tz*(tz>0)
pr=th@w2
print(pr)
