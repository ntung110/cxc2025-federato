import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from transformers import BertModel, BertConfig



class BERT4RecDataset(Dataset):
    def __init__(self, df, max_seq_len=3):
        self.sequences = df["event_type_mapped_list"].to_list()
        self.targets = df["target"].to_list()
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]      # e.g. [10, 22]
        target = self.targets[idx]     # e.g. 15

        # 1) Convert to list if not already
        if not isinstance(seq, list):
            seq = []

        # 2) Pad or truncate to length = max_seq_len
        seq_len = len(seq)
        if seq_len < self.max_seq_len:
            seq = [740] * (self.max_seq_len - seq_len) + seq
        else:
            seq = seq[-self.max_seq_len:]

        # 3) Convert to tensors
        input_seq = torch.tensor(seq, dtype=torch.long)       # shape: (max_seq_len,)
        target_action = torch.tensor(target, dtype=torch.long) # shape: ()

        # 4) Return
        return input_seq, target_action



# Define the BERT4Rec model
class BERT4Rec(nn.Module):
    def __init__(self, num_items, hidden_size, num_layers, num_heads, max_seq_len):
        super(BERT4Rec, self).__init__()
        
        self.config = BertConfig(
            vocab_size=num_items,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_seq_len,
            pad_token_id=740
        )
        
        self.bert = BertModel(self.config)
        self.fc = nn.Linear(hidden_size, num_items)

    def forward(self, input_ids):
        attention_mask = (input_ids != 0).long()
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        last_token_hidden = last_hidden_state[:, -1, :]  # (batch_size, hidden_size)
        logits = self.fc(last_token_hidden)              # (batch_size, num_items)
        return logits
    

def recommend_next_action(model, user_sequence, max_seq_len, device):
    """Predict next-action probabilities for a given user sequence,
       then return top-k predictions and their probabilities."""
    
    model.eval()
    with torch.no_grad():
        # Convert user_sequence to a batched tensor of shape (1, max_seq_len)
        seq_len = len(user_sequence)
        if seq_len < max_seq_len:
            user_sequence = [0]*(max_seq_len - seq_len) + user_sequence
        else:
            user_sequence = user_sequence[-max_seq_len:]

        input_seq = torch.tensor(user_sequence, dtype=torch.long).unsqueeze(0).to(device)
        logits = model(input_seq)  # shape (1, num_items)

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=-1)  # shape (1, num_items)

        # Top-k next items
        top_k = 1
        topk_values, topk_indices = torch.topk(probabilities, k=top_k, dim=-1)  # shape (1, k) each
        topk_values = topk_values.squeeze(0).cpu().numpy()
        topk_indices = topk_indices.squeeze(0).cpu().numpy()

    return topk_indices, topk_values
