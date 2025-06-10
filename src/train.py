import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import EmotionShiftDetector
from dataset import EmotionShiftDataset
import torch.optim as optim
from torch.nn import CrossEntropyLoss

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = CrossEntropyLoss()
    total_loss = 0
    for input_ids, attention_mask, audio_feat, shift_label in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        audio_feat = audio_feat.to(device)
        shift_label = shift_label.to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, audio_feat)
        loss = criterion(logits, shift_label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # TODO: Load your dataset here.
    # For example, conversations = load_your_data()
    # conversations must be a list of list of dicts with keys: 'utterance', 'emotion', optionally 'audio_feat'
    conversations = [
        [   # Example single conversation
            {'utterance': "I'm feeling great today!", 'emotion': 'happy'},
            {'utterance': "Oh, that's awesome to hear.", 'emotion': 'happy'},
            {'utterance': "Actually, I'm a bit worried about tomorrow.", 'emotion': 'sad'},
            {'utterance': "Why's that?", 'emotion': 'neutral'},
            {'utterance': "I have a big presentation.", 'emotion': 'sad'},
        ],
        # Add more conversations...
    ]
    
    dataset = EmotionShiftDataset(conversations, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = EmotionShiftDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 5
    for epoch in range(epochs):
        loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    torch.save(model.state_dict(), 'best_model.pt')
    print("Training complete. Model saved as best_model.pt")

if __name__ == '__main__':
    main()
