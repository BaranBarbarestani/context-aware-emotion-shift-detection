import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import EmotionShiftDetector
from dataset import EmotionShiftDataset

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, audio_feat, shift_label in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            audio_feat = audio_feat.to(device)
            shift_label = shift_label.to(device)
            
            logits = model(input_ids, attention_mask, audio_feat)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == shift_label).sum().item()
            total += shift_label.size(0)
    accuracy = correct / total if total > 0 else 0
    print(f"Evaluation accuracy: {accuracy*100:.2f}%")
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # TODO: Load your test dataset here similar to train.py
    conversations = [
        [   # Example test conversation
            {'utterance': "I'm feeling great today!", 'emotion': 'happy'},
            {'utterance': "Oh, that's awesome to hear.", 'emotion': 'happy'},
            {'utterance': "Actually, I'm a bit worried about tomorrow.", 'emotion': 'sad'},
            {'utterance': "Why's that?", 'emotion': 'neutral'},
            {'utterance': "I have a big presentation.", 'emotion': 'sad'},
        ],
        # Add more test conversations...
    ]
    
    dataset = EmotionShiftDataset(conversations, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8)
    
    model = EmotionShiftDetector().to(device)
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    
    evaluate(model, dataloader, device)

if __name__ == '__main__':
    main()
