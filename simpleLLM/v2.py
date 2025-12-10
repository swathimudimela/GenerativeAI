import torch
import torch.nn as nn
from torch.nn import functional as F

# hyper parameters
batch_size = 32 # how many independent sequence will we process in parallel
block_size = 8 # maximum context length of the predictions
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu' # run on gpu if gpu available
eval_iters = 200
n_embd = 32
# ----------------

torch.manual_seed(1337)

#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

#extract all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
#create mapping from aplha to numeric and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encode charcters to numneric values
decode = lambda l: ''.join([itos[i] for i in l]) # convert numeric values to character strings

# train and text split
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))  # 90% of data
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # we are splitting data into blocksized mini batches
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  # randomly generate batch_size number of offsets and each batch will have blocksize data
    x = torch.stack([data[i : i+block_size] for i in ix])  # create batch_size X block_size matrices
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])  # predictions, for each character at position i prediction will character at i+1
    x, y = x.to(device) , y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):

    """ one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # compute attention scores("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5  
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)

        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out
    

class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        return out
    

class FeedForward(nn.Module):
    """ simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):
    """transformer Block : communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd : embedding dimension, n_head = number of heads we'd like 
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # layer normalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
   
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # embedding vectors based on position
        #self.sa_head = Head(n_embd) # self attention head
        #self.sa_heads = MultiHeadAttention(4, n_embd//4)
        #self.ffwd = FeedForward(n_embd)
        self.blocks = nn.Sequential(
                Block(n_embd, n_head = 4),
                Block(n_embd, n_head = 4),
                Block(n_embd, n_head = 4),
                nn.LayerNorm(n_embd),
        )
        # language model head (lm_head)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        #idx and targets are both (B,T) tensor of integers
        # B : batch size, T : blocksize , C : embedding size
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx) #(B , T, C)
        position_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, C)
        x = token_emb + position_emb # (B, T, C)
        #x = self.sa_heads(x) # apply one head self attention
        #x = self.ffwd(x)
        x = self.blocks(x)
        logits = self.lm_head(x) # decoder language model head (B, T, vocab_size)


        if targets is None:
            loss = None
        else:
            # to calculate loss we will F.crossentropy by cross entropy expects the logits to be [B, C, T] , where as we get [B,T,C] from token_embedding_table
            # to resolve this we change the logits to 2D array and output to 1D array
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the idx to the last block_size tokens
            idx_cond = idx[:,-block_size:]
            #get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:,-1,:] # becomes(B,C)
            # apply the softmax to get probabilities
            probs = F.softmax(logits, dim = -1) # (B,C)
            # sample from th edistribution
            idx_next = torch.multinomial(probs, num_samples = 1) #(B,1)
            #append sample index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B,T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create an optimizer. calculates the gradients and updates the parameters
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
    
    # every once in a while evaluate the loss on the train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss  {losses['train'] :.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaulate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

# generate text from the model
context = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens = 500)[0].tolist()))
