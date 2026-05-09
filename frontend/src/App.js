import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

// Sample reviews dataset (aap apni actual dataset yahan add kar sakte hain)
const SAMPLE_REVIEWS = [
  "Very quick shipping and excellent customer service. Great value for the money!This is a great little",
"Works great, this design makes it easy to hang and it is a nice size for the size of",
"Perfect! Works in your oven and has the right height. I also love that it's removable and",
"Service was great, can't wait to try it out!Very good quality.",
"Arrived quickly and is the perfect size for the long haul. I also have a large one",
"CHEAP CHINA. Not a big deal, but it is a little pricey for a",
"Gave to Mother as a gift. The quality is good but it's not really the best one",
"Great for spices but not for anything else. Not good enough.Nice, sturdy, and functional",
"My car smells no different after 3 weeks. Snake oil.",
"This pillow cushion is well constructed and easily washed. Very comfortable.",
"Haven't used the big ones but the small ones work great!",
"Takes out the mystery of when my meat is finished cooking",
"Its nice to have an organizer large enough for the utensils.",
"Works perfectly. Easy to put in your art and hang.",
"Great little rug! I'd buy it again in different colors.",
"Holds some gift wrap but not longer tubes which is disappointing.",
"Belts never last long enough but this seems to work okay.",
"Such a great price, less expensive than renting & great quality",
"Good for the price but doesn't soften up much after washing",
"I love these sheets!!!! Goes great with my star quilts",
"Great mug - bigger than I thought it would be too!",
"I gave this as a gift and my daughter loves hers.",
"These are great! Great space saver in our laundry room!",
"Very flexible leading edge not quite as helpful under all conditions.",
"bottom will not stay on.....zero customer service do not buy",
"Very nice, elegant glass and does improve the taste of wine.",
"excellent product if you are a side sleeper this is perfect",
"When there not in the oven they make great place mats",
"love it it is so cute in my new bathroom",
"Exactly as pictured! Sturdy and perfect for baking and cooking needs!",
"Top notch artistry done on this bust from grand jesters studios.",
"Love these and I get many comments on this spoon :-)",
"Did not expect this table to be this nice, Thanks",
"Great product. Arrived on time and matched my shower curtain perfectly.",
"a nice secondary vacuum for small apartments and reaching tight areas",
"Good size, good value for money. Looks great in my bar.",
"Somewhat hard to clean, not as convenient as I had hoped.",
"Works Great with all my battery candles of the same brand.",
"Ok pillow kind of too hard and firm not very soft",
"Perfection. Beyond easy to use, minimal effort, fast and perfect froth.",
"Love this item. I smile every time I I use it",
"Fleece blanket is cozy warm. It was exactly what I wanted.",
"Noticed today top I'd bubbling up where a glass was set",
"Great for weighing in lbs and oz not grams. Fast shipping.",
"These were great. I just had to order a longer size.",
"Nice sticks; used these for almost every meal; good for PHO.",
"This is one of the best things I have ever bought",
"Love it but too small. Would love in larger size",
"Perfect for my candy buffet, and it will certainly be multipurpose.",
"Works perfect. Use this for my EV cord in the garage",
"Do not waste your money. It broke after one month.",
"A little tacky, but on the shelf in the spare bathroom.",
"Helps cool my memory mattress a little but not a lot.",
"This is perfect. I wish I would have ordered it earlier.",
"Excellent packaging, Fast shipping , super quality for the right price.",
"Good mix - my favorite is still the Goloka Nag Champa",
"Just what the bride wanted came out nice with the engraving.",
"The color slightly different than I got, the website pictures looks better.",
"Really great, soft and warm. I love the feel of this blanket.",
"LESS THEN 1 YEAR OLD, BUT NO MORE!! I've had mine for 2 months now",
"It's a pint.",
"Complements different color schemes, that are more vibrant and more comfortable.",
"Softer than my more expensive, more expensive, more expensive model, I'm just not sure how",
"I put the bottles of wine in the freezer and they are still fresh! I have one in",
"Small and compact, this travel mug has the design and design of a travel mug.",
"Works like a charm. Item looks great and the materials are good. I have a large one",
"Love it use it daily and it is the best. I will keep my review.Very pretty.",
"Item exactly as described, and it arrived with a very good seal. The dimensions are just",
"does great on pug dog treats. It's a good size and the lid is strong enough",
"Does not fit my husbands bed. It's just too small. I would order one.",
"This is not what I wanted. The quality is just what I wanted. The dimensions are just what",
"Very comfortable and helps my wrists. The only problem is that it's not really a vacuum, but",
"Don't waste your time buying this set, they are just too thin and need to be cleaned every",
"I wish I had bought this sooner, but it is a great product for the price! It",
"Wish these were thicker. The cute polka dots pretty much disappear after baking.",
"The hooks are great for hanging stars from my ceiling! And great price!",
"Returned and exchanged twice now. Base broken every time. Over it.",
"This knife is very sharp, be careful. Worked perfect for what i needed.",
"Nice to travel with, taking them to Italy where ice is rather scarce.",
"If their handles were a bit thicker/stronger they will be a great product."
];

function App() {
  const [mode, setMode] = useState('manual'); // 'manual' or 'dataset'
  const [review, setReview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Dataset browser state
  const [reviews, setReviews] = useState([]);
  const [analyzedReviews, setAnalyzedReviews] = useState(new Map());
  const observerRefs = useRef([]);

  useEffect(() => {
    if (mode === 'dataset') {
      // Initialize reviews with sample data
      const reviewsWithId = SAMPLE_REVIEWS.map((text, index) => ({
        id: index,
        text: text,
        analyzed: false,
        result: null
      }));
      setReviews(reviewsWithId);
    }
  }, [mode]);

  // Intersection Observer for scroll-triggered analysis
  useEffect(() => {
    if (mode !== 'dataset' || reviews.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const reviewId = parseInt(entry.target.dataset.reviewId);
            if (!analyzedReviews.has(reviewId)) {
              analyzeReview(reviewId);
            }
          }
        });
      },
      { threshold: 0.5, rootMargin: '100px' }
    );

    observerRefs.current.forEach((ref) => {
      if (ref) observer.observe(ref);
    });

    return () => observer.disconnect();
  }, [reviews, mode, analyzedReviews]);

  const analyzeReview = async (reviewId) => {
    const review = reviews.find(r => r.id === reviewId);
    if (!review || analyzedReviews.has(reviewId)) return;

    try {
      const response = await axios.post('http://localhost:5000/predict', { 
        review: review.text 
      });
      
      setAnalyzedReviews(prev => new Map(prev).set(reviewId, response.data));
    } catch (error) {
      console.error(`Error analyzing review ${reviewId}:`, error);
    }
  };

  const checkSpam = async () => {
    if (!review.trim()) {
      alert("Please enter a review");
      return;
    }
    
    setLoading(true);
    setResult(null);
    
    try {
      const response = await axios.post('http://localhost:5000/predict', { review });
      setResult(response.data);
    } catch (error) {
      console.error("Error connecting to backend", error);
      alert("Backend server connection failed! Make sure Flask server is running.");
    }
    
    setLoading(false);
  };

  const getResultColor = (resultText) => {
    if (resultText?.includes('Spam')) return '#ff3366';
    if (resultText?.includes('Real')) return '#00ff88';
    return '#ffa500';
  };

  const getConfidenceLevel = (confidence) => {
    const value = parseFloat(confidence);
    if (value >= 80) return 'high';
    if (value >= 60) return 'medium';
    return 'low';
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="header">
        <div className="logo-section">
          <div className="logo-icon">🔍</div>
          <div className="logo-text">
            <h1>SPAM FORENSICS</h1>
            <p className="tagline">AI-Powered Review Authentication System</p>
          </div>
        </div>
        
        <div className="mode-switch">
          <button 
            className={mode === 'manual' ? 'active' : ''}
            onClick={() => setMode('manual')}
          >
            <span>Manual Analysis</span>
          </button>
          <button 
            className={mode === 'dataset' ? 'active' : ''}
            onClick={() => setMode('dataset')}
          >
            <span>Dataset Browser</span>
          </button>
        </div>
      </header>

      {/* Manual Mode */}
      {mode === 'manual' && (
        <div className="manual-mode">
          <div className="input-section">
            <h2>Analyze Review Text</h2>
            <p className="subtitle">Paste any e-commerce review to detect AI-generated spam</p>
            
            <textarea 
              className="review-input"
              rows="8"
              placeholder="Enter review text here..."
              value={review}
              onChange={(e) => setReview(e.target.value)}
            />
            
            <button 
              className="analyze-btn"
              onClick={checkSpam}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Analyzing...
                </>
              ) : (
                <>
                  <span>🔬</span>
                  Run Analysis
                </>
              )}
            </button>
          </div>

          {result && (
            <div className="result-card" style={{ '--result-color': getResultColor(result.result) }}>
              <div className="result-header">
                <h3>Analysis Complete</h3>
                <div className="result-badge" style={{ color: getResultColor(result.result) }}>
                  {result.result}
                </div>
              </div>
              
              <div className="metrics">
                <div className="metric">
                  <span className="metric-label">Confidence Score</span>
                  <div className="confidence-bar">
                    <div 
                      className={`confidence-fill ${getConfidenceLevel(result.confidence)}`}
                      style={{ width: result.confidence }}
                    ></div>
                    <span className="confidence-text">{result.confidence}</span>
                  </div>
                </div>
                
                <div className="metric">
                  <span className="metric-label">Raw Prediction Score</span>
                  <span className="metric-value">{result.raw_score?.toFixed(4)}</span>
                </div>
                
                {result.cleaned_text && (
                  <div className="metric full-width">
                    <span className="metric-label">Processed Text</span>
                    <div className="processed-text">{result.cleaned_text}</div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Dataset Browser Mode */}
      {mode === 'dataset' && (
        <div className="dataset-mode">
          <div className="dataset-header">
            <h2>Review Dataset Browser</h2>
            <p className="subtitle">Scroll through reviews - Analysis happens automatically</p>
            <div className="stats">
              <span className="stat-item">
                Total Reviews: <strong>{reviews.length}</strong>
              </span>
              <span className="stat-item">
                Analyzed: <strong>{analyzedReviews.size}</strong>
              </span>
            </div>
          </div>
          
          <div className="reviews-list">
            {reviews.map((review, index) => {
              const analysis = analyzedReviews.get(review.id);
              
              return (
                <div 
                  key={review.id}
                  ref={el => observerRefs.current[index] = el}
                  data-review-id={review.id}
                  className={`review-item ${analysis ? 'analyzed' : 'pending'}`}
                >
                  <div className="review-number">#{review.id + 1}</div>
                  
                  <div className="review-content">
                    <p className="review-text">{review.text}</p>
                    
                    {analysis ? (
                      <div className="review-analysis">
                        <div className="analysis-badge" style={{ color: getResultColor(analysis.result) }}>
                          {analysis.is_spam ? ' SPAM DETECTED' : '✓ AUTHENTIC'}
                        </div>
                        
                        <div className="mini-metrics">
                          <div className="mini-confidence">
                            <div 
                              className="mini-bar"
                              style={{ 
                                width: analysis.confidence,
                                background: getResultColor(analysis.result)
                              }}
                            ></div>
                            <span>{analysis.confidence}</span>
                          </div>
                          <span className="raw-score">
                            Score: {analysis.raw_score?.toFixed(3)}
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div className="analyzing-indicator">
                        <div className="pulse"></div>
                        <span>Analyzing...</span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="footer">
        <p>Powered by TensorFlow & Deep Learning • Backend: Flask API</p>
      </footer>
    </div>
  );
}

export default App;
