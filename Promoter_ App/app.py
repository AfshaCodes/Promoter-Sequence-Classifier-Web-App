from flask import Flask, render_template, request
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# ------------------------
# CNN Model Definition
# ------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 15, 64)  # (60 / 2 / 2 = 15)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))  # → (B, 32, 30)
        x = self.pool2(torch.relu(self.conv2(x)))  # → (B, 64, 15)
        x = x.view(x.size(0), -1)                  # → (B, 64*15)
        x = torch.relu(self.fc1(x))                # → (B, 64)
        return self.fc2(x)                         # → (B, 2)

# ------------------------
# Load Trained Model
# ------------------------
model = CNNModel()
model.load_state_dict(torch.load("cnn_promoter_model.pth", map_location=torch.device("cpu")))
model.eval()

# ------------------------
# Load Promoter Dataset
# ------------------------
promoter_df = pd.read_csv("All_Promoters.csv")
if 'Sequence' in promoter_df.columns:
    promoter_df = promoter_df.rename(columns={'Sequence': 'sequence'})
promoter_df['sequence'] = promoter_df['sequence'].str.lower()

# ------------------------
# One-Hot Encoding
# ------------------------
def one_hot_encode(seq):
    mapping = {'a':[1,0,0,0], 't':[0,1,0,0], 'c':[0,0,1,0], 'g':[0,0,0,1]}
    return np.array([mapping.get(nuc, [0,0,0,0]) for nuc in seq.lower()])

# ------------------------
# Flask App Setup
# ------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        sequence = request.form["sequence"].strip().lower()
        gene_name = request.form.get("gene", "").strip()

        # Validate sequence length
        if len(sequence) != 60:
            return render_template("result.html", message="❌ Sequence must be exactly 60 bases long.")

        # Encode sequence
        encoded = one_hot_encode(sequence)  # [60, 4]
        input_tensor = torch.tensor(encoded, dtype=torch.float32).T.unsqueeze(0)  # → [1, 4, 60]

        # Predict with CNN
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        # Not a promoter
        if prediction == 0:
            return render_template("result.html", message="❌ This sequence is NOT a promoter.", epd_link="https://epd.epfl.ch")

        # If promoter: check in database
        match = promoter_df[promoter_df['sequence'] == sequence]
        if not match.empty:
            gene_info = match.iloc[0].to_dict()
            return render_template("result.html", message="✅ This is a promoter sequence!", gene_info=gene_info)

        # Not found in local DB but promoter
        elif gene_name:
            epd_url = f"https://epd.epfl.ch/EPDnew_database.php?query={gene_name}&db=human"
            return render_template("result.html", message="✅ Promoter detected, but not in local database.", epd_link=epd_url)

        else:
            return render_template("result.html", message="✅ Promoter detected. Please enter gene name to check EPD.")

    return render_template("index.html")

# ------------------------
# Run the Flask App
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)
