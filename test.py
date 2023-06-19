import torch

# 判断是否有可用的 GPU
if torch.cuda.is_available():
    print('GPU is available.')
    device = torch.device('cuda')
    x = torch.tensor([1.0])
    x = x.to(device)
    print(x)
    y = x ** 2
    print(y)
else:
    print('GPU is not available.')
