import torch.nn as nn
from transformers import BertModel
import torch

class EmotionShiftDetector(nn.Module):
    def __init__(self, audio_feat_dim=13, hidden_dim=128):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.audio_proj = nn.Linear(audio_feat_dim, hidden_dim)
        self.fusion_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self.classifier = nn.Linear(hidden_dim, 2)  # shift/no-shift
        
    def forward(self, input_ids, attention_mask, audio_feat):
        # text encoding: CLS token embedding
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = text_out.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        # project audio features
        audio_emb = self.audio_proj(audio_feat)        # [batch_size, hidden_dim]
        
        # Stack text and audio embeddings as a 2-element sequence for attention
        combined = torch.stack([cls_emb, audio_emb], dim=0)  # [2, batch_size, hidden_dim]
        
        attn_output, _ = self.fusion_attn(combined, combined, combined)
        
        fused_emb = attn_output[0]  # Take fused text embedding [batch_size, hidden_dim]
        
        logits = self.classifier(fused_emb)  # [batch_size, 2]
        return logits
