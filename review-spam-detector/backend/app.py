from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
def clean_text(text):
    """Text ko exactly waise clean karein jaise training mein kiya tha"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
try:
    print("🔄 Loading model and tokenizer...")
    model = tf.keras.models.load_model('spam_model.h5', compile=False)
    
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    try:
        with open('config.pickle', 'rb') as f:
            config = pickle.load(f)
            MAX_LEN = config['max_len']
    except:
        MAX_LEN = 150  
    
    print(f"✅ Model loaded successfully! Using MAX_LEN={MAX_LEN}")
    
except Exception as e:
    print(f"❌ Error loading model/tokenizer: {str(e)}")
    model = None
    tokenizer = None
SAMPLE_REVIEWS = [
    {
        "text": "This product is absolutely amazing! The quality exceeded my expectations and the delivery was super fast. Highly recommend to everyone!",
        "label": "Spam (Likely AI-Generated)"
    },
    {
        "text": "Bought this last week. Works fine. Nothing special but does the job.",
        "label": "Real Review"
    },
    {
        "text": "BEST PURCHASE EVER!!! I am so happy with this product. It changed my life completely. Everyone should buy this right now!!!",
        "label": "Spam (Likely AI-Generated)"
    }
]
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    if model is None or tokenizer is None:
        return jsonify({
            'error': 'Model not loaded properly',
            'status': 'error'
        }), 500
    
    try:
        data = request.json.get('review', '')
        
        if not data or len(data.strip()) < 5:
            return jsonify({
                'error': 'Review text is too short or empty',
                'status': 'error'
            }), 400
        cleaned_text = clean_text(data)
        
        if len(cleaned_text) < 3:
            return jsonify({
                'error': 'Review contains no meaningful text after cleaning',
                'status': 'error'
            }), 400
    
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        prediction_score = float(model.predict(padded, verbose=0)[0][0])
        if prediction_score >= 0.70:
            label = "Spam Review (AI Generated)"
            confidence = prediction_score * 100
            is_spam = True
        elif prediction_score <= 0.35:
            label = "Real Review (Original)"
            confidence = (1 - prediction_score) * 100
            is_spam = False
        else:
            label = "Uncertain - Cannot Determine"
            confidence = 50.0
            is_spam = False
        print("=" * 50)
        print(f"📝 Original Review: {data[:80]}...")
        print(f"🧹 Cleaned Text: {cleaned_text[:80]}...")
        print(f"🎯 Raw Score: {prediction_score:.4f}")
        print(f"🏷️  Label: {label}")
        print(f"📊 Confidence: {confidence:.2f}%")
        print("=" * 50)
        return jsonify({
            'result': label,
            'confidence': f"{confidence:.2f}%",
            'raw_score': prediction_score,
            'is_spam': is_spam,
            'cleaned_text': cleaned_text,
            'status': 'success'
        })
    
    except Exception as e:
        print(f"❌ Prediction Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500
@app.route('/health', methods=['GET'])
def health_check():
    """Check if model is loaded properly"""
    if model is None or tokenizer is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model or tokenizer not loaded'
        }), 500
    
    return jsonify({
        'status': 'healthy',
        'message': 'Spam detector is ready',
        'max_len': MAX_LEN,
        'model_loaded': True
    })
@app.route('/samples', methods=['GET'])
def get_samples():
    """Sample reviews return karein testing ke liye"""
    return jsonify({
        'reviews': SAMPLE_REVIEWS,
        'count': len(SAMPLE_REVIEWS),
        'status': 'success'
    })
@app.route('/predict_batch', methods=['POST', 'OPTIONS'])
def predict_batch():
    """Multiple reviews ek saath predict karein"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model not loaded', 'status': 'error'}), 500
    
    try:
        reviews = request.json.get('reviews', [])
        
        if not reviews or not isinstance(reviews, list):
            return jsonify({'error': 'Invalid input format', 'status': 'error'}), 400
        
        results = []
        
        for review in reviews:
            cleaned = clean_text(review)
            if len(cleaned) < 3:
                results.append({
                    'review': review,
                    'error': 'Too short after cleaning',
                    'status': 'error'
                })
                continue
            
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
            score = float(model.predict(padded, verbose=0)[0][0])
            
            if score >= 0.70:
                label = "Spam Review (AI Generated)"
                confidence = score * 100
                is_spam = True
            elif score <= 0.35:
                label = "Real Review (Original)"
                confidence = (1 - score) * 100
                is_spam = False
            else:
                label = "Uncertain"
                confidence = 50.0
                is_spam = False
            
            results.append({
                'review': review,
                'result': label,
                'confidence': f"{confidence:.2f}%",
                'raw_score': score,
                'is_spam': is_spam,
                'status': 'success'
            })
        
        return jsonify({
            'results': results,
            'count': len(results),
            'status': 'success'
        })
    
    except Exception as e:
        print(f"❌ Batch Prediction Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Spam Detector API Server Starting...")
    print("="*60)
    print(f"📡 Server: http://localhost:5000")
    print(f"🔍 Health Check: http://localhost:5000/health")
    print(f"📝 Sample Reviews: http://localhost:5000/samples")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
