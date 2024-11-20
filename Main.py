import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup # Dont use AdamW under transformers
from torch.optim import AdamW
import matplotlib.pyplot as plt  # For plotting graphs


# Check if a GPU is available and use it, otherwise fallback to CPU
device = torch.device("hip" if torch.cuda.is_available() else "cpu") [1, 2, 3]

print(f"Using device: {device}")

####### DATA PREPARATION ######

# Load and process gene cluster data
def load_gene_clusters(file_path):
    clusters = []
    line_count = 0

    with open(file_path, 'r') as file:
        for line in file:
            line_count += 1
            line = line.strip()  # Strip any leading/trailing spaces
            if not line:
                continue  # Skip empty lines

            # Split the line into components (Cluster ID and genes)
            parts = line.split(',')
            cluster_id = parts[0]  # The first part is the cluster ID
            genes = parts[1:]  # The rest are the gene names

            # Assuming each cluster has an associated function (for example purpose, we'll fake this)
            function = "Gene function for cluster " + cluster_id  # Replace this with actual gene function data
            for gene in genes:
                clusters.append({'cluster_id': cluster_id, 'gene': gene, 'function': function})

    print(f"Lines read: {line_count}")
    # Convert the clusters list into a DataFrame
    return pd.DataFrame(clusters)


# Example usage
gene_info_df = load_gene_clusters('Files/Imported/clusters_gene_info.csv')
print(gene_info_df.head())


####### DATASET PREPARATION FOR BERT ########

class GeneClusterDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256, label_map=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = label_map  # To map gene names to labels (integer encoding)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gene_info = self.data.iloc[idx]
        text = gene_info['function']  # Text to be classified (function)
        label = self.label_map[gene_info['gene']]  # Gene name converted to integer label

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Convert gene names to unique integer labels for classification
label_map = {gene: idx for idx, gene in enumerate(gene_info_df['gene'].unique())}

# Prepare the dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create the dataset for BERT
gene_cluster_dataset = GeneClusterDataset(gene_info_df, tokenizer, label_map=label_map)

# Split into train and test datasets
train_data, test_data = train_test_split(gene_info_df, test_size=0.2, random_state=42)
train_dataset = GeneClusterDataset(train_data, tokenizer, label_map=label_map)
test_dataset = GeneClusterDataset(test_data, tokenizer, label_map=label_map)

# Create DataLoader for the datasets
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)


####### MODEL SETUP FOR SEQUENCE CLASSIFICATION ########

# Initialize the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_map)  # The number of unique labels in the dataset
).to(device)  # Move the model to the correct device (GPU/CPU)

print(f"Model loaded to device: {device}")


###### TRAINING AND EVALUATION ##########

# Set up optimizer, scheduler, and train the model
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3  # Number of training steps (assuming 3 epochs)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# To track the metrics for graph plotting
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []


# Training function (you can adjust this based on your setup)
def train_model(model, train_loader, optimizer, scheduler, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=batch['input_ids'].to(device),
                            attention_mask=batch['attention_mask'].to(device),
                            labels=batch['label'].to(device))

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.logits, dim=1)
            correct_predictions += (predicted == batch['label'].to(device)).sum().item()
            total_predictions += batch['label'].size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        # Record validation metrics at the end of each epoch
        val_loss, val_accuracy = evaluate_model(model, test_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

# Evaluating function
def evaluate_model(model, test_loader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(input_ids=batch['input_ids'].to(device),
                            attention_mask=batch['attention_mask'].to(device),
                            labels=batch['label'].to(device))
            total_loss += outputs.loss.item()
            _, predicted = torch.max(outputs.logits, dim=1)
            correct_predictions += (predicted == batch['label'].to(device)).sum().item()
            total_predictions += batch['label'].size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


# Training the model
train_model(model, train_loader, optimizer, scheduler, epochs=3)

# Plotting the graphs for training and validation loss/accuracy
plt.figure(figsize=(14, 6))

# Plot Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy', color='blue')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()


####### GPT-3 INTEGRATION (Optional) #########
# If you want to use GPT-3 to generate summaries for gene functions, you can integrate the code below.
#
# import openai
#
# openai.api_key = 'your-openai-api-key'
#
# # Example function to generate a GPT-3 summary of a gene function
# def generate_gpt3_summary(gene_name, gene_function):
#     prompt = f"Explain the function of gene {gene_name}: {gene_function}"
#
#     response = openai.Completion.create(
#         engine="text-davinci-003",  # Or use GPT-3.5, whichever is available
#         prompt=prompt,
#         max_tokens=150,
#         temperature=0.7
#     )
#     return response.choices[0].text.strip()
#
# # Example usage for generating GPT-3 summaries
# for idx, row in gene_info_df.iterrows():
#     summary = generate_gpt3_summary(row['gene'], row['function'])
#     print(f"Gene: {row['gene']}, Summary: {summary}")
