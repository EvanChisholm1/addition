import torch
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)

args = parser.parse_args()

chars = ['\n', ' ', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=']

stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for s, i in stoi.items()}

def encode(text):
    return [stoi[c] for c in text]

def decode(toks):
    return ''.join([itos[t] for t in toks])

print('loading model...')
m = torch.load(args.model).to(device)
m.eval()
print('model loaded!')

def solve(equation):
    idx = torch.tensor(encode(equation), dtype=torch.long, device=device)
    while True:
        logits, loss = m(idx.unsqueeze(0))
        idx_next = logits[:, -1, :].argmax(dim=-1).view(1)
        if idx_next.item() == 0: break
        idx = torch.cat((idx,idx_next), dim=-1)

    return decode(idx.tolist())

def add(a, b):
    out = solve(f"\n{a} + {b} =")
    return int(out.split("=")[1][::-1])


while True:
    a = int(input("enter first number: "))
    b = int(input("enter second number: "))

    print(f"solution: {add(a, b)}")

