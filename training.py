import argparse
import torch 
import numpy as np
import torch.nn as nn
import random
import mmap
import torch.nn.functional as F
import pickle

#device

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Training Device: ', device)

##arguments

def parse_args():
    parser = argparse.ArgumentParser(description='Training Arguments')

    parser.add_argument('--batch_size', type = int, required=True, help = 'Provide batch size for training', default=16)
    parser.add_argument('--max_iters', type = int, required=True, help = 'Provide maximum training itterations', default= 1000)
    parser.add_argument('--num_decoder', type = int, required=True, help = 'Provide number of decoder blocks for training', default= 4)
    parser.add_argument('--num_att_heads', type = int, required=True, help = 'Provide number of attention heads for training', default= 4)
    return parser.parse_args()

#hyperparameters


args = parse_args()

n_embedd  = 384 #number of total dims we want to capture from all heads concatenated together
n_decoders = args.num_decoder  #4 decoder layers
dropout = 0.2 #20% neurons are dropped out to prevent overfitting
num_attention_heads= args.num_att_heads # number of attention heads running in parallel
max_iters = args.max_iters  #number of training itterations
learning_rate = 1e-3 #for optimizer
eval_iters = 100 #for reporting loss
block_size = 32  #size of one input sequence
batch_size = args.batch_size  #how many block sized input sequences we want at the same time


#scaled dot product attention

#each head has 96 features and 4 heads in parallel = 384 features

class AttentionHead(nn.Module):

    def __init__(self, head_size):  # head_size-> n_embedd // num_attention_heads  384//4 -> 96
        super().__init__()
        self.key = nn.Linear(n_embedd, head_size, bias = False)  #384 -> 96 features
        self.query = nn.Linear(n_embedd, head_size, bias = False)
        self.value = nn.Linear(n_embedd, head_size, bias = False)
        #register buffer registers no look ahead masking in the model state
        #instaed of reinitializing it for every single head for every single forward and backward pass we add it to model state
        #saves computation
        #efficient way- reduces training time
        #training can be donw without this but it would take longer to complete
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))#(8,8)
        self.dropout = nn.Dropout(dropout) #20% neurons are dropped out

    def forward(self, x):
        #xshape: (B,T,C) --> (4,8,384)
        B,T,C = x.shape  #(4,8,384)
        k = self.key(x) # C or n_embedd (384) -> head_size(96)  (B,T,C) -> (B,T,hs)
        q = self.query(x) # (B,T,C= 384) -> (B,T,hs = 96)
        #calculating attention scores
        #key.transpose(-2,-1) - interchanging last two dims (B,hs,T) = (4, 96, 8)
        #SCALING: *key.shape[-1] = 96 -> sqrt(96) for 
        #query(B,T,hs) @ key(B,hs,T) -> (B, T, T)
        scores = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        #masked fill - repacing 0 to -inf for T timestamp
        #masked fill to prevent look ahead
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        scores = F.softmax(scores, dim = -1) #(B,T,T) #confidence scores for prediction
        scores = self.dropout(scores)  
        v = self.value(x)  #(B,T,hs)
        out = scores @ v
        #print(f'shape after attention head: {out.shape}')
        return out
    

#keys and queries part will be implemented during dot product attention 

class MultiHeadAttention(nn.Module):
    def __init__(self, num_attention_heads, head_size): ##(4, 96)
        super().__init__()
        #bunch of heads in parallel for each head
        self.multiheads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_attention_heads)]) #4 heads running in parallel
        #below transformation adds another set of learnable parameters (weights and bias)
        #kinda helps network learn more about the the text fed
        self.proj = nn.Linear(head_size * num_attention_heads, n_embedd)  #head_size(96) * num_attention_head(4) = 384 ---projected_to---> n_embedd(384)
        self.dropout = nn.Dropout(dropout)  #dropping out 20% of neurons

    def forward(self, x):
        out = torch.cat([head(x) for head in self.multiheads], dim = -1) #concatenation the out each head along the last dimension  (B,T,C)-> concatenate along channel dim
        #4 heads running in parallel each having output (B,T,hs) 
        # on concatenating across dim=-1 output shape = (B,T, 4*hs) -> (B,T,C) 
        out = self.dropout(self.proj(out))
        #print(f'out shape after multi head attention mech: {out.shape}')
        return out 
    

class FeedForward(nn.Module):
    def  __init__(self, n_embedd):
        super().__init__()
        self.netw = nn.Sequential(
            nn.Linear(n_embedd, 4 * n_embedd),
            nn.ReLU(),
            nn.Linear(4 * n_embedd, n_embedd),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        return self.netw(x)
    
class Decoder_block(nn.Module):

    def __init__(self, n_embedd, num_attention_heads):  #input shape to decoder block -> (B, T, C)-(4,8,384)
        super().__init__()
        #n_head is the number of heads and embedd is the embedding dim
        #each head captures 96 features -> head size
        #4 attentions heads working in parallel
        #how many features are each of them capturing?
        head_size = n_embedd // num_attention_heads #is the number of features that each head will be capturing-> 384 // 4 - > 96 head_size
        self.sa = MultiHeadAttention(num_attention_heads, head_size)  #(4, 96)
        self.ffwd = FeedForward(n_embedd) #(384)
        self.ln1 = nn.LayerNorm(n_embedd) #(384) for add and norm
        self.ln2 = nn.LayerNorm(n_embedd) #(384) 

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)  #post norm arch -  Add and Norm. #skip connection 
        y = self.ffwd(x)
        x = self.ln2(x + y)  #residual
        #print(f'shape after Decode block: {x.shape}')
        return x
    
    
class GPTmodel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #what is embedding matrix
        self.embedding_matrix = nn.Embedding(vocab_size, n_embedd)  #char level encoding art to vocab
        self.positional_matrix = nn.Embedding(block_size, n_embedd)  #char level positional embedding wrt to position of each char in a block
        
        self.decoder_blocks = nn.Sequential(*[Decoder_block(n_embedd, num_attention_heads  = num_attention_heads) for _ in range(n_decoders)])
        
        self.final_layer_norm = nn.LayerNorm(n_embedd)
        self.linear_head_final = nn.Linear(n_embedd, vocab_size)   #layer to get the next character given the input characters in hte vocab later softmax will be used to get the most probable character 
        
        #initializing weights with mean = 0.0 and  std dev = 0.02
        #helps training coverge better
        #weights are initialized properly
        self.apply(self._init_weights)

    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                  

    
    def forward(self, index, targets = None):
        #print(index.shape)
        B, T = index.shape

        
        token_embeddings = self.embedding_matrix(index)  ##index (4,8) (B,T) --token_embeddings--> (4,8,384)(B,T,C)
        pos_embeddings = self.positional_matrix(torch.arange(T, device = device))
        x = token_embeddings + pos_embeddings  #broadcasting helps- check broadcasting semantics in torch SHAPE - (4,8,384)
        #print(x.shape)
        x = self.decoder_blocks(x)
        x = self.final_layer_norm(x)
        #print(f'final output shape before linear layer: {x.shape}')
        logits = self.linear_head_final(x)
        #print(f'final output shape: {logits.shape}')

        
        #what is logits?
        #logits = self.embedding_matrix(index)
        if targets is None:
            print('FLAG')
            loss = None
        else:          
            
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            #print(f'shape after reshaping logits: {logits.shape} and targets shape: {targets.shape}')
            loss = F.cross_entropy(logits, targets)
            #print(loss.item())
            
        return logits, loss

    def generate(self,index,max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(index)
            logits = logits[:,-1,:]  #dim ain't dimensioning: it is now as targets are default to None so no loss in logits dim(3-dim)
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples = 1)
            index = torch.cat((index, index_next), dim = 1)
        return index

# model = GPTmodel(vocab_size = len(chars))
# m = model.to(device)

# with open('pickle model/model-02.pkl', 'rb') as f:
#     model = pickle.load(f)
# print('Model parameters loaded successfully')
# m = model.to(device)
# context = torch.zeros((1,1), dtype = torch.long, device= device)
# generated_chars = decode(m.generate(context, max_new_tokens = 500)[0].tolist())
# print(generated_chars)

@torch.no_grad()  #decorator for no grad since we are only caLculating the loss we do not need to compute gradient as no weight updation (optimization) happens here
def estimate_loss(model, encode):
    out = {} 
    model.eval()  #eval model for model evaluation
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batched_data(split, encode)
            logits, loss = model(X, Y)  #model pass without using any gradients
            losses[k] = loss.item()
        out[split] = losses.mean()  #averaging loss over iters
    model.train()  #puts model into training model: weights and biases are updated during this mode
    return out



#get random chunks of data

def get_random_chunks(split, encode):
    filename = "output_train.txt" if split == 'train' else 'output_val.txt'
    with open(filename, 'rb') as f:  #rb- read binary - lot more efficient in binary mode
        with mmap.mmap(f.fileno(), 0, access= mmap.ACCESS_READ) as mm:
            #determining file size and a random position to start reading the Ssnippet of the file
            file_size = len(mm)
            start_position = random.randint(0, (file_size) - block_size*batch_size)
            #print(start_position)
            ##seeking to the random start position and read the block of data
            mm.seek(start_position)  #seeking the random start postion
            block = mm.read(block_size*batch_size-1)  #read this much indexes
            #decode the block to a string, ignoring any invalid byte sequence
            decoded_block = block.decode('utf-8', errors = 'ignore').replace('\r', '')
            #test and train splits
            data = torch.tensor(encode(decoded_block), dtype = torch.long)
    return data

##get batched data

def get_batched_data(split, encode):
    data = get_random_chunks(split, encode)
    random_indices = torch.randint((len(data)-block_size), (batch_size,)) #random indices of batch size inside the data
    #print(random_indices)
    x = torch.stack([data[i:i+block_size]for i in random_indices]) #stacking a batch of 4 input block each of block size = 8 
    y = torch.stack([data[i+1:i+block_size+1]for i in random_indices]) #stacking a batch of 4 target block each of block size = 8 
    return x.to(device), y.to(device)



def main():

    #defining vocabulary
    chars = ""

    with open('vocab.txt', 'r', encoding = 'utf-8') as f:
        text = f.read()
        chars = sorted(set(text))
    vocab_size = len(chars)

    print(f'Vocab size: {vocab_size}')


    ##defining encode and decode text function
    text_to_int = {char:idx for idx, char in enumerate(chars)}
    int_to_text = {idx:char for idx, char in enumerate(chars)}

    encode = lambda z: [text_to_int[char] for char in z]
    decode = lambda z: ''.join(int_to_text[int] for int in z)

    print(f'---------------------Training for---------------------\nblock size: {block_size}, batch_size: {batch_size}, learning rate: {learning_rate}, number of decoder layers: {n_decoders}, number of attention heads operating in parallel: {num_attention_heads}')

    model = GPTmodel(vocab_size = len(chars))
    # m = model.to(device)

    with open('pickle model/model-03.pkl', 'rb') as f:
        model = pickle.load(f)
    print('Model parameters loaded successfully')
    m = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    
    for iter in range(max_iters):
        if iter % eval_iters == 0:
            out = estimate_loss(model = m, encode= encode)
            print(f"Step: {iter}, Training loss: {out['train']:.4f}, Evaluation loss: {out['test']:.4f}")   
        #sample from batch of data
        batched_x, batched_y = get_batched_data(split = 'train', encode = encode)
        #evaluating loss and performing back propagation
        logits, loss = m.forward(batched_x, batched_y)
        optimizer.zero_grad(set_to_none = True)
        loss.backward()  #differentiate the loss fn wrt parameters
        optimizer.step() #update the parameters
    
    print(loss.item()) #.item() get value from torch tensor

    with open('pickle model/model-03.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('model file saved!')

    # print(f'----------------Testing Model----------------')

    # while True:
    #     prompt = str(input('Enter Prompt to generate Model Output: '))
    #     context = torch.tensor(encode(prompt), dtype= torch.long, device= device)
    #     model_output = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    #     print(f'Model Completion:\n{model_output}')



#unsqueeze

if __name__ == '__main__':
    main()