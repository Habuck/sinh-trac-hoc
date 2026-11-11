# What to Watch During CNN Training

## 🎯 Quick Answer

Watch these **4 key metrics** as your model trains through epochs:

1. **Train Loss** - Should steadily **decrease**
2. **Val Loss** - Should steadily **decrease** (but may plateau)
3. **Val Accuracy** - Should steadily **increase**
4. **Gap between Train & Val Loss** - Should stay **small**

---

## 📚 Understanding Key Concepts First

Before diving into what to watch, let's understand the three most important hyperparameters that affect training:

### 1. 🎓 **Learning Rate** - How Big Are the Learning Steps?

**Simple Explanation:**
The learning rate controls **how much** the model adjusts its weights after seeing each batch of data.

**Real-World Analogy:**
Imagine you're trying to find the lowest point in a valley while blindfolded:

- **High Learning Rate (0.01):** You take HUGE steps
  - ✅ Pro: Get to the valley quickly
  - ❌ Con: Might jump over the lowest point and bounce around
  - Like taking 10-meter leaps - fast but imprecise

- **Low Learning Rate (0.0001):** You take tiny steps
  - ✅ Pro: Very precise, won't overshoot
  - ❌ Con: Takes forever to get anywhere
  - Like taking 1-cm steps - slow but accurate

- **Just Right (0.001):** Goldilocks zone
  - ✅ Fast enough to make progress
  - ✅ Small enough to be precise
  - Like taking 1-meter steps - balanced

**What Happens in Practice:**

```python
# Learning Rate = 0.1 (TOO HIGH)
Epoch 1: Loss = 0.6931
Epoch 2: Loss = 0.4521
Epoch 3: Loss = 0.6234  ← Bouncing around!
Epoch 4: Loss = 0.3891
Epoch 5: Loss = 0.5123  ← Can't settle down!

# Learning Rate = 0.001 (JUST RIGHT)
Epoch 1: Loss = 0.6931
Epoch 2: Loss = 0.4521  ← Smooth decrease
Epoch 3: Loss = 0.3456  ← Smooth decrease
Epoch 4: Loss = 0.2891  ← Smooth decrease
Epoch 5: Loss = 0.2456  ← Smooth decrease

# Learning Rate = 0.00001 (TOO LOW)
Epoch 1: Loss = 0.6931
Epoch 2: Loss = 0.6829  ← Barely moving
Epoch 3: Loss = 0.6734  ← Barely moving
Epoch 4: Loss = 0.6645  ← Barely moving
Epoch 5: Loss = 0.6561  ← Too slow!
```

**Common Values:**
- **0.1** - Usually too high for CNNs
- **0.01** - Good starting point for simple models
- **0.001** - Standard default for CNNs (Adam optimizer)
- **0.0001** - Good for fine-tuning or if 0.001 is unstable
- **0.00001** - Usually too low

**How to Know If Your Learning Rate is Wrong:**

| Problem | Learning Rate Issue | Fix |
|---------|-------------------|-----|
| Val loss bouncing up/down wildly | **TOO HIGH** | Reduce by 10x (0.001 → 0.0001) |
| Loss barely decreasing after many epochs | **TOO LOW** | Increase by 10x (0.0001 → 0.001) |
| Loss is NaN or infinity | **WAY TOO HIGH** | Reduce by 100x |
| Smooth decrease, then plateau | **Just right!** | Maybe reduce slightly for fine-tuning |

---

### 2. 📊 **Validation Set** - How Do We Test the Model?

**Simple Explanation:**
The validation set is data the model **never trains on** - it's used to check if the model can handle **new, unseen data**.

**Real-World Analogy:**
Imagine studying for an exam:

- **Training Set:** Practice problems you study from
  - You see these over and over
  - You might memorize the answers
  - Doing well here doesn't guarantee you understand the concepts

- **Validation Set:** Practice exam you take before the real test
  - You've never seen these specific problems
  - Tests if you actually learned the concepts (not just memorized)
  - Tells you if you're ready for the real exam

- **Test Set:** The actual exam (not used during training)
  - Final evaluation
  - Only use once at the very end

**Why We Need It:**

Without a validation set:
```python
# Training on all data
Epoch 10: Train Accuracy = 99.5%  ← Looks amazing!
# But on real alarms: Accuracy = 65%  ← Actually terrible!
```

The model **memorized** the training data instead of learning patterns!

With a validation set:
```python
# Training with validation
Epoch 10: Train Accuracy = 95.2%
Epoch 10: Val Accuracy = 91.8%  ← Close to train = good generalization!
# On real alarms: Accuracy = 90.5%  ← Actually works!
```

**How Your Data is Split:**

```
Your 2,860 audio files:
│
├─ Training Set (50% = 1,430 files)
│  └─ Model learns from these
│     Updates weights based on these
│     Sees these every epoch
│
└─ Validation Set (50% = 1,430 files)
   └─ Model is tested on these
      Never updates weights from these
      Only used to check performance
```

**Why 50/50 Split?**
- Common splits: 80/20, 70/30, 60/40, or 50/50
- You chose 50/50 because:
  - ✅ More validation data = more reliable performance estimate
  - ✅ Still enough training data (1,430 samples is plenty)
  - ✅ Good for medical applications (need reliable validation)

**What Validation Set Tells You:**

| Train Acc | Val Acc | What It Means |
|-----------|---------|---------------|
| 95% | 93% | ✅ **Perfect!** Model generalizes well |
| 95% | 75% | ❌ **Overfitting!** Model memorized training data |
| 65% | 63% | ⚠️ **Underfitting!** Model too simple or needs more training |
| 75% | 85% | 🤔 **Weird!** Check for data leakage or bug |

**Validation Set Size Matters:**

```python
# Small validation set (50 samples)
Epoch 5: Val Acc = 82%
Epoch 6: Val Acc = 76%  ← Big swing from small sample
Epoch 7: Val Acc = 84%  ← Unreliable!

# Large validation set (1,430 samples)
Epoch 5: Val Acc = 82.1%
Epoch 6: Val Acc = 82.8%  ← Stable, reliable
Epoch 7: Val Acc = 83.2%  ← Trustworthy!
```

**Your validation set (1,430 samples) is excellent!** ✅

---

### 3. 📦 **Batch Size** - How Many Samples at Once?

**Simple Explanation:**
Batch size is **how many audio samples** the model looks at before updating its weights.

**Real-World Analogy:**
Imagine you're a teacher grading homework:

- **Batch Size = 1 (Stochastic):** Grade one paper, update your answer key, grade next paper, update again
  - ✅ Pro: Very responsive to each student
  - ❌ Con: Exhausting! Takes forever! Very noisy!
  - Like adjusting your teaching after every single student

- **Batch Size = 1000 (Large Batch):** Grade all papers, then update your answer key once
  - ✅ Pro: Very stable, efficient
  - ❌ Con: Slow to adapt, might miss nuances
  - Like only adjusting teaching once per semester

- **Batch Size = 32 (Mini-Batch):** Grade 32 papers, update, grade next 32, update
  - ✅ Pro: Good balance of speed and stability
  - ✅ Pro: Efficient for GPU computation
  - Like adjusting teaching after each class

**What Happens During Training:**

```python
# Your dataset: 1,430 training samples
# Batch size: 32

# One Epoch = Going through all 1,430 samples once
# Number of batches per epoch = 1,430 / 32 = 44.7 ≈ 45 batches

# What happens in one epoch:
Batch 1:  Process samples 1-32    → Update weights
Batch 2:  Process samples 33-64   → Update weights
Batch 3:  Process samples 65-96   → Update weights
...
Batch 45: Process samples 1409-1430 → Update weights

# Total weight updates per epoch: 45
# Total weight updates for 20 epochs: 45 × 20 = 900 updates
```

**How Batch Size Affects Training:**

| Batch Size | Updates/Epoch | Speed | Stability | Memory |
|------------|---------------|-------|-----------|--------|
| 1 | 1,430 | Slow | Very noisy | Low |
| 8 | 179 | Slow | Noisy | Low |
| 32 | 45 | Medium | Balanced ✅ | Medium |
| 64 | 22 | Fast | Stable | Medium |
| 128 | 11 | Faster | Very stable | High |
| 256 | 6 | Fastest | Too stable? | Very high |

**Visual Example:**

```python
# Batch Size = 8 (Small)
Batch 1:  Loss = 0.6234
Batch 2:  Loss = 0.5891  ← Jumpy
Batch 3:  Loss = 0.6123  ← Jumpy
Batch 4:  Loss = 0.5456  ← Jumpy
Average:  Loss = 0.5926

# Batch Size = 32 (Medium) ← YOUR SETTING
Batch 1:  Loss = 0.6234
Batch 2:  Loss = 0.6012  ← Smoother
Batch 3:  Loss = 0.5891  ← Smoother
Batch 4:  Loss = 0.5734  ← Smoother
Average:  Loss = 0.5968

# Batch Size = 128 (Large)
Batch 1:  Loss = 0.6234
Batch 2:  Loss = 0.6156  ← Very smooth
Batch 3:  Loss = 0.6089  ← Very smooth
Batch 4:  Loss = 0.6012  ← Very smooth
Average:  Loss = 0.6123  ← But slower to decrease!
```

**Common Values:**
- **8-16:** Small batches, noisy but can escape local minima
- **32:** Standard default, good balance ✅ (your current setting)
- **64:** Larger, more stable, faster training
- **128-256:** Very stable, but might need to adjust learning rate
- **512+:** Usually too large for small datasets

**How Batch Size Interacts with Learning Rate:**

```python
# Small batch (8) + High LR (0.01) = Too chaotic!
# Small batch (8) + Low LR (0.0001) = Too slow!

# Medium batch (32) + Medium LR (0.001) = Just right! ✅ (your setting)

# Large batch (128) + Medium LR (0.001) = Might be too conservative
# Large batch (128) + Higher LR (0.01) = Often works well!
```

**Rule of Thumb:**
- Larger batch size → Can use higher learning rate
- Smaller batch size → Need lower learning rate

**Memory Constraints:**

Batch size is limited by your GPU/CPU memory:

```python
# If you get "Out of Memory" error:
BATCH_SIZE = 32  # Try reducing
BATCH_SIZE = 16  # Or even smaller

# If you have lots of memory:
BATCH_SIZE = 64  # Can increase for faster training
BATCH_SIZE = 128
```

**For Your Project:**
- **Batch Size = 32** is perfect for your dataset size (1,430 samples)
- Gives you 45 updates per epoch (good granularity)
- Balanced between speed and stability
- If you see wild fluctuations, try increasing to 64

---

## 🔗 How These Three Work Together

Think of training as driving a car to a destination:

| Parameter | Car Analogy | Effect |
|-----------|-------------|--------|
| **Learning Rate** | How hard you press the gas pedal | Too hard = overshoot turns; too soft = never arrive |
| **Batch Size** | How often you check the GPS | Too often = jerky driving; too rare = miss turns |
| **Validation Set** | Test drive on a different route | Checks if you can actually drive, not just memorized one route |

**Example Combinations:**

```python
# Conservative (slow but safe)
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
# Good for: Fine-tuning, unstable training

# Standard (balanced) ← YOUR CURRENT SETTING
LEARNING_RATE = 0.001
BATCH_SIZE = 32
# Good for: Most cases, default starting point

# Aggressive (fast but risky)
LEARNING_RATE = 0.01
BATCH_SIZE = 128
# Good for: Large datasets, simple problems
```

---

## 📊 What You'll See During Training

### Example Output (Each Epoch):

```
Epoch [ 1/20] | Train Loss: 0.6234 | Val Loss: 0.5891 | Val Acc: 68.50%
Epoch [ 2/20] | Train Loss: 0.4521 | Val Loss: 0.4234 | Val Acc: 78.20%
Epoch [ 3/20] | Train Loss: 0.3456 | Val Loss: 0.3821 | Val Acc: 82.10%
Epoch [ 4/20] | Train Loss: 0.2891 | Val Loss: 0.3512 | Val Acc: 84.30%
...
Epoch [20/20] | Train Loss: 0.0823 | Val Loss: 0.2145 | Val Acc: 91.20%
```

---

## ✅ Good Signs (What You WANT to See)

### 1. **Steady Decrease in Train Loss**
```
Epoch [ 1/20] | Train Loss: 0.6234
Epoch [ 2/20] | Train Loss: 0.4521  ✓ Going down
Epoch [ 3/20] | Train Loss: 0.3456  ✓ Going down
Epoch [ 4/20] | Train Loss: 0.2891  ✓ Going down
```
**Meaning:** Model is learning from the training data

### 2. **Steady Decrease in Val Loss**
```
Epoch [ 1/20] | Val Loss: 0.5891
Epoch [ 2/20] | Val Loss: 0.4234  ✓ Going down
Epoch [ 3/20] | Val Loss: 0.3821  ✓ Going down
Epoch [ 4/20] | Val Loss: 0.3512  ✓ Going down
```
**Meaning:** Model is generalizing well to unseen data

### 3. **Steady Increase in Val Accuracy**
```
Epoch [ 1/20] | Val Acc: 68.50%
Epoch [ 2/20] | Val Acc: 78.20%  ✓ Going up
Epoch [ 3/20] | Val Acc: 82.10%  ✓ Going up
Epoch [ 4/20] | Val Acc: 84.30%  ✓ Going up
```
**Meaning:** Model is getting better at predictions

### 4. **Small Gap Between Train and Val Loss**
```
Epoch [10/20] | Train Loss: 0.1523 | Val Loss: 0.1891  ✓ Gap = 0.037 (small)
Epoch [15/20] | Train Loss: 0.1012 | Val Loss: 0.1456  ✓ Gap = 0.044 (small)
```
**Meaning:** Model is not overfitting

---

## ⚠️ Warning Signs (What to Watch Out For)

### 1. **Overfitting** (Most Common Problem)

**What it looks like:**
```
Epoch [10/20] | Train Loss: 0.1523 | Val Loss: 0.1891  ✓ Small gap
Epoch [11/20] | Train Loss: 0.1234 | Val Loss: 0.1923  ⚠️ Val loss going UP
Epoch [12/20] | Train Loss: 0.0987 | Val Loss: 0.2145  ⚠️ Val loss still going UP
Epoch [13/20] | Train Loss: 0.0756 | Val Loss: 0.2389  ❌ Gap getting BIGGER
```

**What it means:**
- Model is **memorizing** training data instead of learning patterns
- Model performs great on training data but **poorly on new data**
- **This is BAD for real-world use!**

**What to do:**
- ✅ Stop training (use model from epoch 10)
- ✅ Add data augmentation
- ✅ Reduce model complexity
- ✅ Add dropout layers

### 2. **Underfitting**

**What it looks like:**
```
Epoch [ 1/20] | Train Loss: 0.6234 | Val Loss: 0.6123
Epoch [ 5/20] | Train Loss: 0.5891 | Val Loss: 0.5823  ⚠️ Barely improving
Epoch [10/20] | Train Loss: 0.5645 | Val Loss: 0.5512  ⚠️ Still barely improving
Epoch [20/20] | Train Loss: 0.5234 | Val Loss: 0.5123  ❌ Stuck at high loss
```

**What it means:**
- Model is **too simple** to learn the patterns
- Model performs **poorly** on both training and validation data

**What to do:**
- ✅ Train for more epochs
- ✅ Increase model complexity (more layers/filters)
- ✅ Decrease learning rate
- ✅ Check if data has enough signal

### 3. **Validation Loss Fluctuating Wildly**

**What it looks like:**
```
Epoch [ 8/20] | Val Loss: 0.2345
Epoch [ 9/20] | Val Loss: 0.4123  ⚠️ Jumped up
Epoch [10/20] | Val Loss: 0.1987  ⚠️ Jumped down
Epoch [11/20] | Val Loss: 0.3654  ⚠️ Jumped up again
```

**What it means:**
- Learning rate might be **too high**
- Validation set might be **too small**
- Model is unstable

**What to do:**
- ✅ Reduce learning rate (e.g., from 0.001 to 0.0001)
- ✅ Use learning rate scheduler
- ✅ Increase batch size

### 4. **No Improvement at All**

**What it looks like:**
```
Epoch [ 1/20] | Train Loss: 0.6931 | Val Acc: 50.00%
Epoch [ 5/20] | Train Loss: 0.6931 | Val Acc: 50.00%  ⚠️ No change
Epoch [10/20] | Train Loss: 0.6931 | Val Acc: 50.00%  ⚠️ No change
Epoch [20/20] | Train Loss: 0.6931 | Val Acc: 50.00%  ❌ Stuck!
```

**What it means:**
- Model is **not learning at all**
- Loss of 0.6931 and 50% accuracy = **random guessing**

**What to do:**
- ✅ Check if data is loaded correctly
- ✅ Check if labels are correct
- ✅ Increase learning rate
- ✅ Check model architecture

---

## 🎢 Special Case: Fluctuating Validation Loss

### What You're Seeing:
```
Epoch [ 8/20] | Val Loss: 0.2345
Epoch [ 9/20] | Val Loss: 0.4123  ⚠️ Jumped up!
Epoch [10/20] | Val Loss: 0.1987  ⚠️ Jumped down!
Epoch [11/20] | Val Loss: 0.3654  ⚠️ Jumped up again!
Epoch [12/20] | Val Loss: 0.2156  ⚠️ Jumped down again!
```

### 🔍 Root Causes (Using Concepts Above)

#### Cause 1: **Learning Rate Too High** (Most Common)

**What's happening:**
Remember the "walking down a valley" analogy? Your steps are too big!

```
Optimal point: ★

With LR = 0.001 (too high for this model):
Epoch 8:  You're here → ●
Epoch 9:  Big step → ●        (overshot!)
Epoch 10: Big step → ●    (overshot back!)
Epoch 11: Big step →     ●   (overshot again!)

With LR = 0.0001 (just right):
Epoch 8:  You're here → ●
Epoch 9:  Small step → ●
Epoch 10: Small step → ●
Epoch 11: Small step → ★  (reached optimal!)
```

**How to diagnose:**
- Large fluctuations (±30% or more)
- No clear downward trend
- Training loss is smooth but val loss is wild

**Solution:**
```python
# In train_cnn_window_split.ipynb, change:
LEARNING_RATE = 0.001  # Current

# To:
LEARNING_RATE = 0.0001  # 10x smaller
```

**Expected result:**
```
Before (LR=0.001):
Epoch 8:  Val Loss: 0.2345
Epoch 9:  Val Loss: 0.4123  ← Wild swing
Epoch 10: Val Loss: 0.1987  ← Wild swing

After (LR=0.0001):
Epoch 8:  Val Loss: 0.2345
Epoch 9:  Val Loss: 0.2289  ← Smooth
Epoch 10: Val Loss: 0.2234  ← Smooth
```

---

#### Cause 2: **Batch Size Too Small**

**What's happening:**
Remember: smaller batches = noisier updates!

```python
# With BATCH_SIZE = 8 (small)
# Each batch might be very different:
Batch 1: [alarm, alarm, alarm, silence, silence, alarm, silence, alarm]
         → Loss focuses on alarms
Batch 2: [silence, silence, silence, silence, alarm, silence, silence, silence]
         → Loss focuses on silence
Batch 3: [alarm, silence, alarm, alarm, silence, alarm, alarm, silence]
         → Balanced
# Result: Weights bounce around!

# With BATCH_SIZE = 64 (larger)
# Each batch is more representative:
Batch 1: [32 alarms, 32 silences] → Balanced
Batch 2: [31 alarms, 33 silences] → Balanced
Batch 3: [33 alarms, 31 silences] → Balanced
# Result: Stable weight updates!
```

**How to diagnose:**
- Current batch size < 32
- Fluctuations are moderate (±10-20%)
- Both train and val loss fluctuate

**Solution:**
```python
# In train_cnn_window_split.ipynb, change:
BATCH_SIZE = 32  # Current

# To:
BATCH_SIZE = 64  # 2x larger
# Or even:
BATCH_SIZE = 128  # 4x larger (if memory allows)
```

**Trade-off:**
- ✅ More stable training
- ✅ Faster computation (fewer batches)
- ❌ Might need to increase learning rate slightly
- ❌ Uses more memory

---

#### Cause 3: **Validation Set Too Small**

**What's happening:**
Remember: small validation set = unreliable estimates!

```python
# Small validation set (50 samples)
# If model gets 3 more wrong:
Epoch 8:  42/50 correct = 84.0% accuracy
Epoch 9:  39/50 correct = 78.0% accuracy  ← 6% drop from 3 samples!

# Large validation set (1,430 samples)
# If model gets 3 more wrong:
Epoch 8:  1,200/1,430 correct = 83.92% accuracy
Epoch 9:  1,197/1,430 correct = 83.71% accuracy  ← 0.21% drop (stable!)
```

**How to diagnose:**
```python
# Check your validation set size:
ls datasets/dataset_w1.5s_h0.25s_20260314/val/*.wav | wc -l
```

**Your validation set:** 1,430 samples ✅ **This is excellent!**

If you had < 200 samples, that would be a problem. But you don't!

---

#### Cause 4: **Model on Edge of Overfitting**

**What's happening:**
The model is trying to memorize training data, causing instability:

```
Epoch 8:  Model learns general patterns → Val loss: 0.2345 ✅
Epoch 9:  Model starts memorizing training data → Val loss: 0.4123 ❌
Epoch 10: Weights adjust back → Val loss: 0.1987 ✅
Epoch 11: Memorizing again → Val loss: 0.3654 ❌
```

**How to diagnose:**
- Train loss keeps decreasing smoothly
- Val loss bounces up and down
- Gap between train and val loss is growing

**Solution:**
- ✅ Stop training earlier (use best epoch)
- ✅ Add dropout layers
- ✅ Reduce learning rate
- ✅ Add data augmentation

---

### 📊 How Much Fluctuation is Normal?

#### ✅ **Acceptable Fluctuation:**
```
Epoch [ 8/20] | Val Loss: 0.2345
Epoch [ 9/20] | Val Loss: 0.2456  ← +4.7% (small)
Epoch [10/20] | Val Loss: 0.2287  ← -6.9% (small)
Epoch [11/20] | Val Loss: 0.2398  ← +4.9% (small)
```
**Variation:** ±5-10% is normal noise

**What to do:** Nothing! This is fine. Look at the overall trend.

---

#### ⚠️ **Moderate Fluctuation:**
```
Epoch [ 8/20] | Val Loss: 0.2345
Epoch [ 9/20] | Val Loss: 0.2891  ← +23% (moderate)
Epoch [10/20] | Val Loss: 0.2156  ← -25% (moderate)
Epoch [11/20] | Val Loss: 0.2734  ← +27% (moderate)
```
**Variation:** ±15-30% is concerning

**What to do:**
1. Reduce learning rate by 5-10x
2. Increase batch size to 64 or 128

---

#### ❌ **Severe Fluctuation:**
```
Epoch [ 8/20] | Val Loss: 0.2345
Epoch [ 9/20] | Val Loss: 0.5123  ← +118% (severe!)
Epoch [10/20] | Val Loss: 0.1234  ← -76% (severe!)
Epoch [11/20] | Val Loss: 0.4567  ← +270% (severe!)
```
**Variation:** ±50%+ is a serious problem

**What to do:**
1. **Immediately** reduce learning rate by 10-100x
2. Check for bugs in data loading
3. Verify labels are correct
4. Consider restarting training

---

### 🛠️ Step-by-Step Fix Guide

**Step 1: Identify the fluctuation severity**

Calculate the variation:
```python
# Look at your output:
Epoch 8:  Val Loss: 0.2345
Epoch 9:  Val Loss: 0.4123

# Calculate change:
Change = (0.4123 - 0.2345) / 0.2345 = 0.758 = 75.8% increase

# Severity:
< 10%  → Normal ✅
10-30% → Moderate ⚠️
> 30%  → Severe ❌
```

**Step 2: Apply fixes in order**

```python
# Fix #1: Reduce Learning Rate (try this first!)
LEARNING_RATE = 0.001  # Change to:
LEARNING_RATE = 0.0001  # If moderate fluctuation
LEARNING_RATE = 0.00001 # If severe fluctuation

# Fix #2: Increase Batch Size (if Fix #1 doesn't help)
BATCH_SIZE = 32   # Change to:
BATCH_SIZE = 64   # Or:
BATCH_SIZE = 128

# Fix #3: Combine both
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
```

**Step 3: Retrain and compare**

```
Before fixes:
Epoch 8:  Val Loss: 0.2345
Epoch 9:  Val Loss: 0.4123  ← 75% jump
Epoch 10: Val Loss: 0.1987  ← 52% drop

After fixes (LR=0.0001, BS=64):
Epoch 8:  Val Loss: 0.2345
Epoch 9:  Val Loss: 0.2289  ← 2.4% drop ✅
Epoch 10: Val Loss: 0.2234  ← 2.4% drop ✅
```

---

### 📈 Focus on the Trend, Not Individual Epochs

Even with fluctuations, what matters is the **overall trend**:

#### ✅ Good (despite noise):
```
Epoch [ 1/20] | Val Loss: 0.5891
Epoch [ 5/20] | Val Loss: 0.3821  ← Trend: DOWN
Epoch [10/20] | Val Loss: 0.2456  ← Trend: DOWN
Epoch [15/20] | Val Loss: 0.1987  ← Trend: DOWN
Epoch [20/20] | Val Loss: 0.1756  ← Trend: DOWN
```
**Overall:** Going from 0.59 → 0.18 (70% improvement) ✅

Even if individual epochs bounce around, the model is learning!

#### ❌ Bad (no trend):
```
Epoch [ 1/20] | Val Loss: 0.5891
Epoch [ 5/20] | Val Loss: 0.4821
Epoch [10/20] | Val Loss: 0.5234  ← Trend: FLAT
Epoch [15/20] | Val Loss: 0.4567
Epoch [20/20] | Val Loss: 0.5123  ← Trend: FLAT
```
**Overall:** Stuck around 0.50 (no improvement) ❌

The model isn't learning - fix the hyperparameters!

---

### 💡 Pro Tip: Use a Moving Average

Instead of looking at individual epochs, calculate a moving average:

```python
# Raw values (noisy):
Epoch 8:  Val Loss: 0.2345
Epoch 9:  Val Loss: 0.4123  ← Spike!
Epoch 10: Val Loss: 0.1987
Epoch 11: Val Loss: 0.3654  ← Spike!
Epoch 12: Val Loss: 0.2156

# 3-epoch moving average (smooth):
Epoch 8:  Avg(6,7,8)   = 0.2456
Epoch 9:  Avg(7,8,9)   = 0.2938  ← Smoother
Epoch 10: Avg(8,9,10)  = 0.2818  ← Smoother
Epoch 11: Avg(9,10,11) = 0.3255  ← Smoother
Epoch 12: Avg(10,11,12) = 0.2599 ← Smoother
```

The moving average shows the true trend more clearly!

---

## 📈 Ideal Training Curve

### Perfect Scenario:
```
Epoch [ 1/20] | Train: 0.6234 | Val: 0.5891 | Val Acc: 68.50%
Epoch [ 2/20] | Train: 0.4521 | Val: 0.4234 | Val Acc: 78.20%  ✓
Epoch [ 3/20] | Train: 0.3456 | Val: 0.3821 | Val Acc: 82.10%  ✓
Epoch [ 4/20] | Train: 0.2891 | Val: 0.3512 | Val Acc: 84.30%  ✓
Epoch [ 5/20] | Train: 0.2456 | Val: 0.3234 | Val Acc: 85.80%  ✓
...
Epoch [15/20] | Train: 0.0923 | Val: 0.2145 | Val Acc: 91.20%  ✓
Epoch [16/20] | Train: 0.0891 | Val: 0.2156 | Val Acc: 91.15%  ⚠️ Val plateauing
Epoch [17/20] | Train: 0.0856 | Val: 0.2167 | Val Acc: 91.10%  ⚠️ Val getting worse
Epoch [18/20] | Train: 0.0823 | Val: 0.2189 | Val Acc: 91.05%  ❌ STOP HERE!
```

**Best model:** Epoch 15 (before validation started getting worse)

---

## 🎯 For Your Glucose Alarm Project

### What's Most Important: **RECALL** (Sensitivity)

After training completes, check the final metrics:

```
Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.89      0.92       287
           1       0.90      0.96      0.93       287  ← THIS IS MOST IMPORTANT!
```

**For medical alarms, you want:**
- ✅ **Recall (Class 1) > 95%** - Don't miss any alarms!
- ✅ Precision (Class 1) > 85% - Minimize false alarms
- ⚠️ It's OK if precision is lower - **missing an alarm is worse than a false alarm**

---

## 📊 After Training: Check the Plots

The notebook will show you plots. Look for:

### 1. **Loss Curves**
- Train loss should be smooth and decreasing
- Val loss should decrease then plateau
- Gap between them should be small

### 2. **Accuracy Curve**
- Should steadily increase
- Should plateau near the end

### 3. **Confusion Matrix**
```
Predicted:    0    1
Actual:
    0       255   32   ← False Positives (32) - OK
    1        12  275   ← False Negatives (12) - MINIMIZE THIS!
```

**For alarms:**
- **False Negatives (bottom-left)** = Missed alarms ❌ **VERY BAD**
- **False Positives (top-right)** = False alarms ⚠️ **Annoying but OK**

---

## ✅ Quick Checklist While Training

- [ ] Train loss decreasing?
- [ ] Val loss decreasing?
- [ ] Val accuracy increasing?
- [ ] Gap between train/val loss small (< 0.1)?
- [ ] No wild fluctuations?
- [ ] Training time reasonable (not stuck)?

If all ✅ → **Training is going well!**

If any ❌ → **Check the warning signs above**

---

## 💡 Pro Tips

1. **Don't panic if val loss fluctuates a bit** - Small fluctuations are normal
2. **The best model is NOT always the last epoch** - Use the epoch with lowest val loss
3. **For medical alarms, prioritize RECALL over accuracy** - Missing an alarm is dangerous!
4. **Compare multiple window sizes** - 1.0s vs 1.5s vs 2.0s to find the best

---

## 🎓 Summary

**Good training looks like:**
- 📉 Losses going down
- 📈 Accuracy going up
- 🤏 Small gap between train and val
- 🎯 High recall on final metrics

**Bad training looks like:**
- 📈 Val loss going up (overfitting)
- 📊 Stuck at same values (not learning)
- 🎢 Wild fluctuations (unstable)
- 🎯 Low recall on final metrics

**Your goal:** Get **recall > 95%** for Class 1 (alarms) to ensure no medical alarms are missed!

---

## 📋 Quick Reference: Hyperparameter Cheat Sheet

### Learning Rate Quick Guide

| Value | When to Use | Pros | Cons |
|-------|-------------|------|------|
| **0.1** | Almost never for CNNs | Very fast | Too unstable, will diverge |
| **0.01** | Simple models, large batches | Fast convergence | Often too high for CNNs |
| **0.001** | **Default starting point** ✅ | Good balance | May need tuning |
| **0.0001** | Fine-tuning, unstable training | Very stable | Slower convergence |
| **0.00001** | Transfer learning final layers | Maximum stability | Very slow |

**Your current setting:** `0.001` ✅

**If you see fluctuating val loss:** Try `0.0001`

---

### Batch Size Quick Guide

| Value | Updates/Epoch* | When to Use | Memory | Stability |
|-------|----------------|-------------|--------|-----------|
| **8** | 179 | Debugging, very small datasets | Low | Noisy |
| **16** | 89 | Small datasets | Low | Somewhat noisy |
| **32** | 45 | **Default, most cases** ✅ | Medium | Balanced |
| **64** | 22 | Larger datasets, more stability | Medium | Stable |
| **128** | 11 | Large datasets, fast training | High | Very stable |
| **256** | 6 | Very large datasets | Very high | Extremely stable |

*Based on your dataset size of 1,430 training samples

**Your current setting:** `32` ✅

**If you see fluctuating val loss:** Try `64` or `128`

---

### Validation Set Quick Guide

| Size | Reliability | When to Use |
|------|-------------|-------------|
| **< 100** | ❌ Unreliable | Avoid if possible |
| **100-200** | ⚠️ Somewhat reliable | Only if data is limited |
| **200-500** | ✅ Reliable | Good for most cases |
| **500-1000** | ✅ Very reliable | Excellent |
| **> 1000** | ✅ Extremely reliable | Ideal for medical applications |

**Your current validation set:** `1,430 samples` ✅ **Excellent!**

---

### Common Problem → Solution Matrix

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Val loss bouncing ±30%+ | Learning rate too high | Reduce LR: `0.001` → `0.0001` |
| Val loss bouncing ±10-20% | Batch size too small | Increase BS: `32` → `64` |
| Loss barely decreasing | Learning rate too low | Increase LR: `0.0001` → `0.001` |
| Train loss ↓, Val loss ↑ | Overfitting | Stop early, add dropout, reduce LR |
| Both losses stuck high | Underfitting | More epochs, bigger model, higher LR |
| Loss = NaN or infinity | Learning rate way too high | Reduce LR by 100x, restart |
| Loss = 0.6931, Acc = 50% | Not learning at all | Check data/labels, increase LR |
| Smooth train, wild val | Learning rate too high | Reduce LR: `0.001` → `0.0001` |

---

### Recommended Settings for Your Project

#### Conservative (Stable, Slower)
```python
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
NUM_EPOCHS = 30
```
**Use when:** You see fluctuating validation loss

#### Balanced (Default) ✅
```python
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 20
```
**Use when:** Starting fresh, standard training

#### Aggressive (Fast, Riskier)
```python
LEARNING_RATE = 0.01
BATCH_SIZE = 128
NUM_EPOCHS = 15
```
**Use when:** You need quick experiments, have lots of data

---

## 🎯 Final Checklist

Before you start training:
- [ ] Selected the right dataset (1.0s vs 1.5s vs 2.0s)
- [ ] Set learning rate (start with `0.001`)
- [ ] Set batch size (start with `32`)
- [ ] Set number of epochs (start with `20`)
- [ ] Understand what to watch for during training

During training:
- [ ] Train loss decreasing?
- [ ] Val loss decreasing (overall trend)?
- [ ] Val accuracy increasing?
- [ ] Gap between train/val loss < 0.1?
- [ ] Fluctuations < 10%?

After training:
- [ ] Check final recall for Class 1 (alarms)
- [ ] Goal: Recall > 95% ✅
- [ ] Check confusion matrix (minimize false negatives!)
- [ ] Save the model with best validation loss

---

**You're now fully equipped to train your CNN and understand what's happening!** 🚀

If you see anything unexpected, refer back to this guide to diagnose and fix it!

