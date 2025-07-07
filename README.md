
# 🧬 Promoter Sequence Classifier

This project is a deep learning-based DNA promoter sequence classifier built using PyTorch and deployed as a Flask web application. It determines whether a 60bp DNA sequence is a promoter or not and provides gene-level details using a local database and the EPD website.

---

## 📁 Project Structure

```
Promoter_Classifier/
│
├── final_dataset.csv           # Combined promoter/non-promoter dataset
├── All_Promoters.csv           # Scraped EPD promoter data
├── cnn_promoter_model.pth      # Trained CNN model
├── app.py                      # Flask application code
├── templates/
│   ├── index.html              # Input form
│   └── result.html             # Result display
└── static/                     # CSS/images (optional)
```

---

## 🚀 Features

- Predicts promoter or non-promoter from a 60bp DNA sequence.
- Searches local EPD-based promoter database.
- Optionally asks for gene name to retrieve info from EPD.
- Offers 3 trained models: CNN, BiLSTM, Transformer.
- Web interface built using Flask + Bootstrap.

---

## 🧠 Deep Learning Models Used

- **CNN**: Used in web app. Fast + accurate (~90% accuracy).
- **BiLSTM**: Sequence learning using bidirectional LSTMs.
- **Transformer**: Attention-based DNA modeling.

---

## 🔧 How to Run

1. Clone the repo.
2. Install dependencies:
```bash
pip install flask torch pandas numpy
```
3. Run the Flask app:
```bash
python app.py
```
4. Open in browser: `http://127.0.0.1:5000`

---

## 🛡 License

This project is shared under **CC BY-NC 4.0** license for **educational/non-commercial use only**. Please cite properly and do not use commercially without permission.

---

## 👩‍💻 Author

**Afshan Azeem**  
Genomic AI Researcher | Bioinformatics | Deep Learning  
GitHub: [your-profile]  
Date: July 07, 2025
