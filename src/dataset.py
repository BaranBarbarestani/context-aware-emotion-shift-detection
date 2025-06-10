import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class EmotionShiftDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_len=128):
        """
        conversations: list of conversations,
            each conversation is a list of dicts with keys:
            'utterance': str,
            'emotion': str,
            optionally 'audio_feat': torch.Tensor of shape (13,)
        tokenizer: transformers tokenizer instance
        """
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Map emotions to integer IDs
        self.emotion2id = {'neutral':0, 'happy':1, 'sad':2, 'angry':3}
        
        self.data = []
        for conv in conversations:
            prev_emotion = None
            for turn in conv:
                emotion_id = self.emotion2id.get(turn['emotion'], 0)
                shift = 0 if prev_emotion is None else int(prev_emotion != emotion_id)
                prev_emotion = emotion_id
                
                self.data.append({
                    'text': turn['utterance'],
                    'audio_feat': turn.get('audio_feat', torch.zeros(13)),
                    'shift_label': shift,
                    'emotion_id': emotion_id
                })
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)       # [max_len]
        attention_mask = encoding['attention_mask'].squeeze(0)  # [max_len]
        audio_feat = item['audio_feat'].float()            # [13]
        shift_label = torch.tensor(item['shift_label'], dtype=torch.long)
        return input_ids, attention_mask, audio_feat, shift_label
