# 🚀 QUICK START GUIDE

## Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 2: Check Your Dataset
```bash
python check_dataset.py
```

**Ye dekho:**
- Class distribution balanced hai?
- Reviews ki average length kitni hai?
- Fake aur Real mein kya difference hai?

## Step 3: Train Model
```bash
python train_improved.py
```

**Training time:** 10-30 minutes (dataset size pe depend karta hai)

**Training ke baad check karo:**
```
=== CLASSIFICATION REPORT ===
              precision    recall  f1-score
   Real (OR)       X.XX      X.XX      X.XX
   Fake (CG)       X.XX      X.XX      X.XX
```

**Target Metrics (Good Model):**
- Precision (Fake): >0.80
- Recall (Fake): >0.80
- F1-Score (Fake): >0.80

**Agar ye numbers nahi mil rahe:**
- Dataset mein problem hai
- More/better data collect karo

## Step 4: Test Model
```bash
python test_model.py
```

Ye tumhare sample reviews pe model test karega.

## Step 5: Run API Server
```bash
python app_improved.py
```

Server start ho jayega: `http://localhost:5000`

## Step 6: Test API
```bash
# Terminal mein ye command run karo
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "Amazing product!!! Best ever!!!"}'
```

## Step 7: Connect Frontend
React mein ye code use karo:
```javascript
const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ review: userInput })
});

const data = await response.json();
console.log(data.result);        // "Real Review" ya "Spam Review"
console.log(data.confidence);    // "85.34%"
console.log(data.raw_score);     // 0.8534
```

---

## 🔧 Troubleshooting

### Problem: "Model sab kuch Real bol raha"
**Solution:**
1. `check_dataset.py` run karo - class distribution dekho
2. Agar Fake reviews kam hain, more data collect karo
3. `app_improved.py` mein line 76 pe threshold 0.4 kar do

### Problem: "Model sab kuch Fake bol raha"
**Solution:**
1. Threshold 0.6 kar do
2. Dataset mein Real reviews ki quality check karo

### Problem: "Low confidence (50-60%)"
**Solution:**
1. Model properly train nahi hua
2. More epochs chahiye
3. Better/more data chahiye

### Problem: "Module not found error"
**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: "File not found: spam_model.h5"
**Solution:**
Pehle model train karo:
```bash
python train_improved.py
```

---

## 📊 Files Explained

| File | Purpose |
|------|---------|
| `check_dataset.py` | Dataset quality check |
| `train_improved.py` | Model training |
| `test_model.py` | Quick model testing |
| `app_improved.py` | Flask API server |
| `spam_model.h5` | Trained model (created after training) |
| `tokenizer.pickle` | Text tokenizer (created after training) |
| `config.pickle` | Settings (created after training) |
| `requirements.txt` | Python dependencies |

---

## 💡 Pro Tips

1. **Dataset Quality > Model Complexity**
   - Achi data = Acha model
   - 10,000 good reviews > 50,000 bad reviews

2. **Monitor Training**
   - `training_history.png` dekho
   - Validation loss badhna = overfitting

3. **Test Regularly**
   - Har change ke baad `test_model.py` run karo
   - Sample reviews se manually verify karo

4. **Backup**
   ```bash
   cp spam_model.h5 backups/spam_model_$(date +%Y%m%d).h5
   ```

5. **Production Deployment**
   - Debug mode off karo (`debug=False`)
   - Gunicorn use karo instead of Flask dev server
   - CORS properly configure karo

---

## ✅ Checklist Before Going Live

- [ ] Dataset quality checked (`check_dataset.py`)
- [ ] Model trained successfully
- [ ] Classification report shows good metrics (>80%)
- [ ] Test samples working correctly (`test_model.py`)
- [ ] API endpoints tested
- [ ] Frontend connected and working
- [ ] Error handling tested
- [ ] Model backup created

---

**Need Help?**
Run `python check_dataset.py` aur output share karo!
