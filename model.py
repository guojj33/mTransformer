import torch
import torch.nn as nn
import torch.nn.functional as F

class mMLP(nn.Module):
    def __init__(self, embed_dim, intermediate_dim):
        super().__init__()
        self.ln1 = nn.Linear(embed_dim, intermediate_dim)
        self.ln2 = nn.Linear(intermediate_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.ln2(hidden_states)
        return hidden_states
    
class mAttention(nn.Module):
    '''
    multi-head scaled-dot-product attention with casual mask
    '''
    def __init__(self, embed_dim):
        super().__init__()
        # attention mask
        self.embed_dim = embed_dim
        self.num_head = 12
        self.head_dim = self.embed_dim // self.num_head
        self.ln_attn = nn.Linear(embed_dim, 3*embed_dim)

    def split_heads(self, tensor):
        new_shape = tensor.shape[:-1] + (self.num_head, self.head_dim)
        tensor = tensor.view(new_shape).permute(0, 2, 1, 3)
        return tensor # (batch, num_head, seq_length, head_dim)

    def forward(self, hidden_states, layer_past=None):
        query, key, value = self.ln_attn(hidden_states).split(self.embed_dim, -1)    # [batch, seq_len, embed_dim]
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=True) # (batch, num_head, seq_len, head_dim)
        bsz, _, seq_len, _ = attn_output.shape
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)   # (batch, seq_len, embed_dim)
        present = (key, value)
        return attn_output, present

class mBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = mMLP(embed_dim, 4*embed_dim)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = mAttention(embed_dim)

    def forward(self, hidden_states, layer_past=None):
        residual = hidden_states
        # norm
        hidden_states = self.ln_1(hidden_states)
        # multi-head attention
        attn_output, present = self.attn(hidden_states, layer_past=layer_past)
        # residual connection
        hidden_states =  attn_output + residual

        residual = hidden_states
        # norm
        hidden_states = self.ln_2(hidden_states)
        # feed forward
        feed_forward_output = self.mlp(hidden_states)
        # residual connection
        hidden_states = feed_forward_output + residual
        return hidden_states, present

class mTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed_dim = 768
        self.vocab_size = vocab_size
        self.max_position_embedding = 1024
        self.num_hidden_layers = 12
        self.num_head = 12

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(self.max_position_embedding, self.embed_dim)

        self.h = nn.ModuleList([mBlock(self.embed_dim) for i in range(self.num_hidden_layers)])

        self.ln = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

    def forward(self, input_ids, past_key_value=None, position_ids=None):
        # input_ids [bz, seq_len]
        # to embdes
        inputs_embeds = self.wte(input_ids) # [bz, seq_len, embed_dim]
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=input_ids.device).unsqueeze(0) # [bz, seq_len]
        position_embeds = self.wpe(position_ids)    # [bz, seq_len, embed_dim]
        hidden_states = inputs_embeds + position_embeds

        presents = ()   # key value cache
        for i in range(self.num_hidden_layers):
            block = self.h[i]
            layer_past = None
            if past_key_value is not None:
                layer_past = past_key_value[i]
            hidden_states, present = block(hidden_states, layer_past=layer_past)
            presents = presents + (present,)

        logits = self.ln(hidden_states) # [bz, seq_len, vocab_size]
        return hidden_states, logits, presents