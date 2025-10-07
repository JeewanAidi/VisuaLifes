# ❓ VisuaLife FAQ (Frequently Asked Questions)

Welcome to the **VisuaLife FAQ** — here you’ll find solutions to the most common questions and issues faced by developers and researchers using VisuaLife.

---

##  1. Installation & Setup

###  Q1: How do I install VisuaLife?
Clone the repo and install dependencies:
```bash
git clone https://github.com/<your-username>/VisuaLife.git
cd VisuaLife
pip install -r requirements.txt
```
You can test the setup by running:

```bash
pytest tests/
```

###  Q2: I get ModuleNotFoundError: No module named 'visualife'
Make sure you are running scripts from the project root.
If running from **experiments/**, add the parent path manually:

```python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

###  Q3: How do I run an experiment?
Use any of the example scripts inside the experiments/ directory.
**Example:**

```bash
python experiments/mlp_train_mnist.py
```

For CNN or webcam demos:

```bash
python experiments/webcam_demo.py
```

##  2. Model Training
### Q4: My model is training too slowly. Why?

You can try the following:

- Reduce the batch size.
- Lower the number of epochs for early testing.
- Use smaller layers (e.g., `64 → 32 → 10` instead of `512 → 256 → 10`).
- Avoid unnecessary print statements inside training loops.
- Use NumPy vectorized operations wherever possible.

###  Q5: How can I monitor loss and accuracy?
You can add callback functions or print metrics each epoch:

```python
for epoch in range(epochs):
    loss = model.fit(X_train, Y_train)
    print(f"Epoch {epoch+1} | Loss: {loss:.4f}")
```
Or use a custom callback in [visualife/core/callbacks.py](visualife/core/callbacks.py) for live logs or saving checkpoints.

###  Q6: How do I save and load my trained model?
VisuaLife supports model serialization using pickle:

```python
import pickle

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

##  3. Layers, Activations & Losses
### Q7: What activation functions are supported?
Currently supported activations:

- ReLU
- Sigmoid
- Tanh
- Softmax

Defined in: [visualife/core/activations.py](visualife/core/activations.py)

### Q8: What loss functions are included?
- MeanSquaredError
- CrossEntropyLoss

You can add more in [visualife/core/losses.py](visualife/core/losses.py) and document them in [docs/Apis/utils.md](docs/Apis/utils.md).


### Q9: Can I create my own layer?
Yes! Create a custom class in visualife/core/ like this:

```python
class Dropout:
    def __init__(self, rate=0.5):
        self.rate = rate

    def forward(self, x):
        self.mask = (np.random.rand(*x.shape) > self.rate)
        return x * self.mask

    def backward(self, grad_output, learning_rate):
        return grad_output * self.mask

```
Then import and use:

```python
from visualife.core.dropout import Dropout
```

##  4. Data Handling
### Q10: How do I load a custom image dataset?
Use the DataLoader from visualife/utils/data_loader.py:

```python
from visualife.utils.data_loader import DataLoader

loader = DataLoader("dataset/", img_size=(64,64), batch_size=16)
for X, y in loader:
    print(X.shape, y.shape)
```
This automatically detects classes based on folder names and returns normalized image batches.

### Q11: My DataLoader skips some images.
That’s expected if an image is corrupted or unreadable.
You’ll see a warning like:

```arduino
 Skipping /path/image.jpg, error: cannot identify image file
```
Check and re-save that image manually.

## 5. Debugging & Development
### Q12: How do I test new modules?
All test files are inside /tests/.
Example to test new layer logic:

```bash
pytest tests/test_layers.py -v
```

### Q13: My gradients are NaN or exploding

Possible reasons:

- Learning rate too high → try reducing to `0.0005`
- Weight initialization too large
- Missing gradient clipping

You can implement gradient clipping in your optimizer easily:

```python
gradients = np.clip(gradients, -1, 1)

## 6. Educational Use & Expansion
### Q14: Is VisuaLife suitable for teaching?

Absolutely!  
VisuaLife was built as a didactic framework — it’s minimal, transparent, and perfect for:

- Understanding forward/backward propagation
- Implementing CNNs, MLPs, or RNNs manually
- Learning the math behind deep learning

### Q15: Can VisuaLife be extended for research?

Yes.  
It is modular and open-source — you can:

- Plug in custom optimizers
- Experiment with hybrid CNN + Transformer layers
- Add datasets for vision or NLP
- Integrate with PyTorch/TensorFlow for hybrid testing

## 7. Community & Support
### Q16: How can I report a bug or request a feature?

Open an issue on GitHub:

[https://github.com/<your-username>/VisuaLife/issues](https://github.com/<your-username>/VisuaLife/issues)

When reporting, provide:

- Python version
- Full error traceback
- Steps to reproduce


### Q17: Who maintains VisuaLife?
Jeewan Aidi (Farwestern University, SOE) — with contributions from peers and the open-source community.
If you’re interested in joining development, see docs/contributing.md.

## Final Note
VisuaLife was created not just as a framework — but as a learning revolution 
It’s your window to understand how deep learning truly works — one layer, one gradient, one neuron at a time.

© 2025 VisuaLife | Built with by Jeewan Aidi