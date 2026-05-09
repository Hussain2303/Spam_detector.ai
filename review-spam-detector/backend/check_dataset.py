import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

print("="*60)
print("📊 SPAM DATASET QUALITY ANALYSIS")
print("="*60)

# ============================================
# 1. LOAD DATASET
# ============================================
try:
    df = pd.read_csv('fake reviews dataset.csv')
    print(f"\n✅ Dataset loaded successfully!")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
except Exception as e:
    print(f"\n❌ Error loading dataset: {e}")
    exit(1)

# ============================================
# 2. BASIC STATISTICS
# ============================================
print("\n" + "="*60)
print("📈 BASIC STATISTICS")
print("="*60)

print(f"\nTotal Reviews: {len(df)}")
print(f"Missing Values:\n{df.isnull().sum()}")

# ============================================
# 3. CLASS DISTRIBUTION (MOST IMPORTANT!)
# ============================================
print("\n" + "="*60)
print("🏷️  CLASS DISTRIBUTION")
print("="*60)

label_counts = df['label'].value_counts()
print(f"\n{label_counts}")

cg_count = label_counts.get('CG', 0)
or_count = label_counts.get('OR', 0)
total = cg_count + or_count

if total > 0:
    print(f"\nFake (CG): {cg_count} ({cg_count/total*100:.1f}%)")
    print(f"Real (OR): {or_count} ({or_count/total*100:.1f}%)")
    
    # Check balance
    ratio = max(cg_count, or_count) / min(cg_count, or_count) if min(cg_count, or_count) > 0 else 0
    
    if ratio > 3:
        print(f"\n⚠️  WARNING: Dataset is HIGHLY IMBALANCED! (Ratio: {ratio:.1f}:1)")
        print("   - This is why your model might be biased!")
        print("   - Consider collecting more data for minority class")
        print("   - Or use techniques like SMOTE for balancing")
    elif ratio > 1.5:
        print(f"\n⚠️  Dataset is somewhat imbalanced (Ratio: {ratio:.1f}:1)")
        print("   - Class weights will help (already implemented)")
    else:
        print(f"\n✅ Dataset is well balanced! (Ratio: {ratio:.1f}:1)")

# ============================================
# 4. TEXT LENGTH ANALYSIS
# ============================================
print("\n" + "="*60)
print("📏 TEXT LENGTH ANALYSIS")
print("="*60)

df['text_length'] = df['text_'].astype(str).str.len()
df['word_count'] = df['text_'].astype(str).str.split().str.len()

print(f"\nOverall Statistics:")
print(f"  Average length: {df['text_length'].mean():.1f} characters")
print(f"  Average words: {df['word_count'].mean():.1f} words")
print(f"  Min length: {df['text_length'].min()}")
print(f"  Max length: {df['text_length'].max()}")

# By class
print(f"\nBy Class:")
for label in ['CG', 'OR']:
    if label in df['label'].values:
        subset = df[df['label'] == label]
        print(f"\n  {label} Reviews:")
        print(f"    Avg length: {subset['text_length'].mean():.1f} chars")
        print(f"    Avg words: {subset['word_count'].mean():.1f} words")

# Check for very short reviews
short_reviews = df[df['word_count'] < 3]
if len(short_reviews) > 0:
    print(f"\n⚠️  {len(short_reviews)} reviews have less than 3 words (might be low quality)")

# ============================================
# 5. COMMON WORDS ANALYSIS
# ============================================
print("\n" + "="*60)
print("💬 COMMON WORDS IN EACH CLASS")
print("="*60)

def get_common_words(texts, n=15):
    """Extract most common words"""
    all_words = []
    for text in texts:
        # Simple cleaning
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        # Remove very common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'it', 'this', 'that', 'with', 'as', 'was', 'i', 'my'}
        words = [w for w in words if w not in stop_words and len(w) > 2]
        all_words.extend(words)
    return Counter(all_words).most_common(n)

for label in ['CG', 'OR']:
    if label in df['label'].values:
        subset = df[df['label'] == label]['text_'].values
        common = get_common_words(subset, 10)
        
        label_name = "Fake (CG)" if label == 'CG' else "Real (OR)"
        print(f"\nTop words in {label_name}:")
        for word, count in common:
            print(f"  • {word}: {count}")

# ============================================
# 6. SPECIAL CHARACTERS ANALYSIS
# ============================================
print("\n" + "="*60)
print("❗ SPECIAL CHARACTERS USAGE")
print("="*60)

df['exclamation_count'] = df['text_'].astype(str).str.count('!')
df['question_count'] = df['text_'].astype(str).str.count('\?')
df['caps_words'] = df['text_'].astype(str).str.findall(r'\b[A-Z]{2,}\b').str.len()

print("\nBy Class:")
for label in ['CG', 'OR']:
    if label in df['label'].values:
        subset = df[df['label'] == label]
        print(f"\n  {label} Reviews:")
        print(f"    Avg exclamations: {subset['exclamation_count'].mean():.2f}")
        print(f"    Avg questions: {subset['question_count'].mean():.2f}")
        print(f"    Avg CAPS words: {subset['caps_words'].mean():.2f}")
print("\n" + "="*60)
print("📝 SAMPLE REVIEWS")
print("="*60)

print("\n5 Random FAKE (CG) Reviews:")
print("-" * 60)
fake_samples = df[df['label'] == 'CG'].sample(min(5, len(df[df['label'] == 'CG'])))
for idx, row in fake_samples.iterrows():
    print(f"• {row['text_'][:100]}...")

print("\n5 Random REAL (OR) Reviews:")
print("-" * 60)
real_samples = df[df['label'] == 'OR'].sample(min(5, len(df[df['label'] == 'OR'])))
for idx, row in real_samples.iterrows():
    print(f"• {row['text_'][:100]}...")
print("\n" + "="*60)
print("💡 RECOMMENDATIONS")
print("="*60)

recommendations = []

# Check dataset size
if len(df) < 1000:
    recommendations.append("⚠️  Dataset too small! Collect at least 5000-10000 reviews for good accuracy")
elif len(df) < 5000:
    recommendations.append("⚠️  Dataset size is okay but more data would improve performance")
else:
    recommendations.append("✅ Dataset size is good")

# Check balance
if ratio > 3:
    recommendations.append("⚠️  Fix class imbalance: collect more minority class data")
elif ratio > 1.5:
    recommendations.append("⚡ Slight imbalance present, but class weights should handle it")
else:
    recommendations.append("✅ Classes are well balanced")

avg_words = df['word_count'].mean()
if avg_words < 5:
    recommendations.append("⚠️  Reviews are too short! Model needs more context")
elif avg_words < 10:
    recommendations.append("⚡ Reviews are short but workable")
else:
    recommendations.append("✅ Review length is good")

# Check diversity
unique_ratio = len(df['text_'].unique()) / len(df)
if unique_ratio < 0.5:
    recommendations.append("⚠️  Many duplicate reviews detected! This hurts model generalization")
else:
    recommendations.append("✅ Good variety in reviews")

print()
for rec in recommendations:
    print(rec)
try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Class distribution
    axes[0, 0].bar(label_counts.index, label_counts.values, color=['#ff6b6b', '#51cf66'])
    axes[0, 0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(label_counts.values):
        axes[0, 0].text(i, v + 50, str(v), ha='center', fontweight='bold')
    df.boxplot(column='text_length', by='label', ax=axes[0, 1])
    axes[0, 1].set_title('Text Length by Class', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Characters')
    df.boxplot(column='word_count', by='label', ax=axes[1, 0])
    axes[1, 0].set_title('Word Count by Class', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Words')
    df.boxplot(column='exclamation_count', by='label', ax=axes[1, 1])
    axes[1, 1].set_title('Exclamation Marks by Class', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Count')
    
    plt.suptitle('Dataset Quality Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'dataset_analysis.png'")
except Exception as e:
    print(f"\n⚠️  Could not create visualizations: {e}")

print("\n" + "="*60)
print("✅ ANALYSIS COMPLETE!")
print("="*60)
print("\nNext Steps:")
print("1. Review the warnings and recommendations above")
print("2. If data quality is good, run: python train_improved.py")
print("3. Check training results and classification report")
print("4. If accuracy is low, collect more/better data")
print("="*60)
