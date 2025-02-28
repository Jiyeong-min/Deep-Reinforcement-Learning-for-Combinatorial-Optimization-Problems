import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention layer
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.W1 = nn.Linear(self.hidden_dim,self.hidden_dim,bias=False)
        self.W2 = nn.Linear(self.hidden_dim,self.hidden_dim,bias=False)
        self.V = nn.Linear(self.hidden_dim,1,bias=False)
        self.tanh = nn.Tanh()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
    
    def forward(self, enc_outputs, dec_output, mask):
        w1_e = self.W1(enc_outputs)
        w2_d = self.W2(dec_output)
        tanh_output = self.tanh(w1_e + w2_d)
        v_dot_tanh = self.V(tanh_output).squeeze(2)
        # masking
        v_dot_tanh += mask
        attention_weights = F.softmax(v_dot_tanh, dim=1)
        return attention_weights
    

class Encoder(nn.Module):
    """
    Encoder in PtrNet
    """
    def __init__(self, hidden_dim, input_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        self.cell = nn.GRU(self.input_dim, self.hidden_dim, 1,batch_first=True)
            
    def forward(self, input):
        enc_output, enc_hidden_state = self.cell(input)
        return enc_output, enc_hidden_state
          

class Decoder(nn.Module):
    """
    Decoder in PtrNet
    """
    
    def __init__(self, hidden_dim, input_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        self.cell = nn.GRU(self.input_dim, self.hidden_dim, 1,batch_first=True)
        self.attention_layer = Attention(self.hidden_dim)
                
    def forward(self, input, enc_output, hidden_state, pointer, mask):
        """
        pointer: (batch_size, 1)
        """
        idx = pointer.repeat(1,2).unsqueeze(1)
        dec_output, dec_hidden = self.cell(input.gather(1,idx),hidden_state)
        attention_weights = self.attention_layer(enc_output,dec_output,mask)

        return attention_weights, dec_hidden


class PtrNet(nn.Module):
    
    def __init__(self, hidden_dim, input_dim=2, deterministic=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.deterministic = deterministic
        
        self.encoder = Encoder(self.hidden_dim)
        self.decoder = Decoder(self.hidden_dim)
    
    def forward(self, input):
        batch_size = input.size(0)
        seq_len = input.size(1)
        if input.is_cuda:
            probs = torch.zeros(batch_size,1,device=torch.device('cuda'))
            pointers = torch.zeros(batch_size,1,dtype=torch.long,device=torch.device('cuda'))
            mask = torch.zeros(batch_size,seq_len,dtype=torch.float,device=torch.device('cuda'))
            
            pointer = torch.zeros(batch_size,1,dtype=torch.long,device=torch.device('cuda'))
        
        else:
            probs = torch.ones(batch_size,1)
            pointers = torch.zeros(batch_size,1)
            mask = torch.zeros(batch_size,seq_len)
            
            pointer = torch.zeros(batch_size,1,dtype=torch.long)

        # Encoding
        enc_output, enc_hidden_state = self.encoder(input)
        
        mask = self.update_mask(mask,pointer)
        
        # Decoding
        for i in range(seq_len-1):
            if i == 0:
                attention_weights, dec_hidden_state = self.decoder(input, enc_output, enc_hidden_state, pointer, mask)
            else:
                attention_weights, dec_hidden_state = self.decoder(input, enc_output, dec_hidden_state, pointer, mask)
            
            
            if self.deterministic:
                prob, pointer = torch.max(attention_weights, dim=1)
                mask = self.update_mask(mask,pointer)
                prob = prob.unsqueeze(1)
                pointer = pointer.unsqueeze(1)
            else:
                pointer = attention_weights.multinomial(1, replacement=True)
                prob = torch.gather(attention_weights,1,pointer)
                mask = self.update_mask(mask,pointer)       
            
            probs += torch.log(prob)
            pointers = torch.cat([pointers,pointer],dim=1)
        return probs, pointers
    
    def update_mask(self,mask,pointer):
        for batch,i in enumerate(pointer):
            mask[batch,i] = float('-inf')
        return mask
    
    def get_length(self, input, solution):
             
        current_coords = torch.gather(input, 1, solution.unsqueeze(-1).expand(-1, -1, 2))       
        next_coords = torch.roll(current_coords, -1, dims=1)       
        distances = torch.sqrt(torch.sum((current_coords - next_coords) ** 2, dim=-1))       
        tour_length = torch.sum(distances, dim=1)
        
        return tour_length.unsqueeze(1)
        
class Critic(nn.Module):
    
    def __init__(self, hidden_dim, input_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        self.encoder = Encoder(self.hidden_dim)
        self.decoder_1 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.decoder_2 = nn.Linear(self.hidden_dim,1)
        self.relu = nn.ReLU()
    
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
    
    def forward(self, input):
        enc_output, enc_hidden_state = self.encoder(input)
        dec_hidden_state = self.decoder_1(enc_hidden_state)
        dec_hidden_state = self.relu(dec_hidden_state)
        dec_output = self.decoder_2(dec_hidden_state)
        return dec_output.squeeze(0)
        