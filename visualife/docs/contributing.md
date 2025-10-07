#  Contributing to VisuaLife

Welcome to **VisuaLife** — a deep learning framework built from scratch for AI education and assistive vision projects.  
We’re excited that you want to contribute! 🎉

---

##  1. How to Contribute

You can help improve VisuaLife in many ways:

-  Add new layers or activation functions  
-  Implement optimizers, metrics, or loss functions  
-  Improve documentation or tutorials  
-  Add new test cases under `tests/`  
-  Report or fix bugs  

| Area | Description |
|------|--------------|
|  **Code** | Add new layers, optimizers, or utilities |
|  **Docs** | Improve tutorials, fix typos, or add examples |
|  **Tests** | Write or improve test cases for modules |
|  **Feedback** | Suggest ideas or report issues |


---

##  2. Setting Up Your Environment

**Clone the repository and install dependencies:**
```bash
git clone https://github.com/<your-username>/VisuaLife.git
cd VisuaLife
```

**Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run tests to verify setup:**
```bash
pytest tests/
```
Test that everything works:

##  3. Repository Structure
```bash
visualife/
 ├── core/               # Core framework (layers, models, optimizers, etc.)
 ├── utils/              # Utilities (data loader, data collector)
 ├── docs/               # Documentation (API + Tutorials)
 ├── tests/              # Unit tests for all components
 └── experiments/        # Example training scripts and demos
```

## 4. Code Style Guidelines

Please follow a consistent and readable style:

- Use `snake_case` for variables and functions
- Use `CamelCase` for class names
- Add **docstrings** to all public methods
- Keep line length under **100 characters**
- Use **type hints** wherever possible
- Comment complex math or logic clearly

**Example:**

```python
class Dense:
    """Fully connected layer"""

    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

```

## 5. Adding a New Layer

If you want to add a custom layer:

- Create a new file under `visualife/core/` or add it inside `layers.py`
- Define your class (e.g., `Dropout`, `BatchNorm`, etc.)
- Implement these methods:
  - `forward(inputs)`
  - `backward(grad_output, learning_rate)`
- Add test cases under `tests/test_layers.py`
- Document it in `docs/Apis/layers.md`

**Example Skeleton:**

```python
class Dropout:
    def __init__(self, rate=0.5):
        self.rate = rate

    def forward(self, inputs):
        self.mask = (np.random.rand(*inputs.shape) > self.rate).astype(np.float32)
        return inputs * self.mask

    def backward(self, grad_output, learning_rate):
        return grad_output * self.mask

```

## 6. Writing Documentation

Every new feature should have:

- API documentation (`docs/Apis/*.md`)
- Example usage (`docs/Tutorials/*.md`)

Follow the same Markdown format as:

```shell
# Title
> Short description

## Example
```python
# Your code here


```markdown
## 🧪 7. Running Tests

Before submitting, run all tests:

```bash
pytest -v
```
Or run a specific file:

```bash
pytest tests/test_layers.py
```

## 8. Submitting Pull Requests
**Create a new branch:**

```bash
git checkout -b feature-new-layer
```
**Commit your changes:**

```bash
git commit -m "Added Dropout layer and tests"
```
**Push and open a PR:**

```bash
git push origin feature-new-layer
```

## Good First Contributions

If you’re new, start with easy tasks:

- Improve docstrings or examples
- Add small test cases in `tests/`
- Fix simple bugs or typos
- Write short tutorials

## 9. Community

If you face issues, feel free to:

- Open an issue on GitHub
- Join our discussion on new features
- Share your experiments in the community tab


## 10. Acknowledgement

VisuaLife is maintained by Jeewan Aidi and contributors at
Farwestern University — Department of Engineering (SOE)

Contributors and students are always welcome! 
Your support helps make AI education open and accessible for everyone.
Together, we can make AI more accessible, transparent, and educational.

© 2025 VisuaLife | Created and maintained by Jeewan Aidi