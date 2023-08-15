import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

!wget -O names.txt 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'  # replace with your URL

with open('names.txt', 'r') as f:
    words = f.read().splitlines()

chars = sorted(list(set(''.join(words))))
strToInt = {s:i+1 for i,s in enumerate(chars)}
strToInt['.'] = 0
intToStr = {i:s for s,i in strToInt.items()}
vocab_size = len(intToStr)

block_size = 3 #context length/ how many chars are taken as input to predict next char?

def build_dataset(words):
  X, Y = [], []

  for w in words:
    context = [0] * block_size
    for char in w + '.':
     ix = strToInt[char]
     X.append(context)
     Y.append(ix)
     context = context[1:] + [ix]
  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X,Y

#create training, dev, and test subsets of the dataset
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

#utility func to compare manual gradients to PyTorch grads
def cmp(s, dt, t):
  ex = torch.all(dt == t.grad).item()
  app = torch.allclose(dt, t.grad)
  maxdiff = (dt - t.grad).abs().max().item()
  print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')


n_embd = 10
n_hidden = 64

C = torch.randn((vocab_size, n_embd))
W1 = torch.randn((n_embd * block_size, n_hidden)) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden) * 0.1

W2 = torch.randn((n_hidden, vocab_size)) * 0.1
b2 = torch.randn(vocab_size) * 0.1

bngain = torch.randn((1, n_hidden)) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden)) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters))
for p in parameters:
  p.requires_grad = True

batch_size = 32
n = batch_size 
# construct a minibatch
ix = torch.randint(0, Xtr.shape[0], (batch_size,))
Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

#forward pass

emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)
hprebn = embcat @ W1 + b1 #torch.Size([32, 64])

#batchnorm layer
bnmeani = 1/n*hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bndiff * bnraw + bnbias

#non-linearity
h = torch.tanh(hpreact)

#2nd linear layer
logits = h @ W2 + b2

#cross entropy loss
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

#PyTorch backward pass
for p in parameters:
  p.grad = None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv,
          norm_logits, logit_maxes, logits, h, hpreact, bnraw, 
          bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
          embcat, emb]:
  t.retain_grad()
loss.backward() 

dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n
dprobs = (1/probs) * dlogprobs

#because counts_sum_inv is broadcast when mult. with counts, that array needs to be collapsed column-wise, and those gradients added together
dcounts_sum_inv = (counts * dprobs).sum(1, keepdims=True) 
dcounts = counts_sum_inv * dprobs
dcounts_sum = -counts_sum**-2 * dcounts_sum_inv
dcounts += torch.ones_like(counts) * dcounts_sum
dnorm_logits = norm_logits.exp() * dcounts
dlogits = dnorm_logits.clone()
dlogit_maxes = -dnorm_logits.sum(1, keepdims=True)

#compare manual gradients to torch grads
cmp('logprobs', dlogprobs, logprobs)
cmp('probs', dprobs, probs)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)
cmp('counts_sum', dcounts_sum, counts_sum)
cmp('counts', dcounts, counts)
cmp('norm_logits', dnorm_logits, norm_logits)
cmp('logit_maxes', dlogit_maxes, logit_maxes)
