# Installation Guide

> Learn how to install and set up **VisuaLife**, your custom-built deep learning framework from scratch.

---

## ⚙️ Prerequisites

Before installing, ensure your environment meets the following requirements:

| Requirement | Recommended Version |
|--------------|---------------------|
| **Python** | 3.9 or higher |
| **pip** | 21.0 or higher |
| **Operating System** | Windows / Linux / macOS |
| **Dependencies** | NumPy, Matplotlib (for visualization), tqdm (for progress bars) |

---

##  1. Clone the Repository

First, clone the official **VisuaLife** repository from GitHub:

```bash
git clone https://github.com/JeewanAidi/VisuaLifes.git
cd VisuaLifes
```
This will create a folder structure like:
VisuaLifes/
├── visualife/
├── docs/
├── examples/
├── tests/
└── README.md

##  2. Set Up a Virtual Environment (Recommended)
It’s good practice to create an isolated environment for your project:

🪟 On Windows

```bash
python -m venv venv
venv\Scripts\activate
```

🐧 On macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

##  3. Install Dependencies
Install the required dependencies for running VisuaLife:


```bash
pip install -r requirements.txt
```
If you plan to build the documentation website, install the documentation dependencies too:

```bash
pip install -r requirements-docs.txt
```
The requirements-docs.txt file includes MkDocs and plugins for generating a professional website version of the docs.


##  4. Verify Installation
To check if everything is set up correctly, run a quick Python test:


```bash
python
```
Then inside the Python shell:
```python
from visualife.core import layers
print("✅ VisuaLife imported successfully!")
```
If you see the message above, your installation is complete 🎉

##  5. Run Example
To quickly verify the framework in action, run an included example:

```bash
python examples/basic_network_demo.py
```
Expected output:
```yaml
Training Epoch 1/5
Loss: 0.3456  |  Accuracy: 89.2%
```

##  6. Optional: Build the Documentation Site (MkDocs)
VisuaLife uses MkDocs for generating its documentation website.

To build and preview locally:

```bash
mkdocs serve
```
Then open your browser and go to:

http://127.0.0.1:8000

You’ll see your documentation website live — just like TensorFlow’s docs.

##  Troubleshooting

| Problem | Possible Fix |
|---------|--------------|
| ModuleNotFoundError: No module named 'visualife' | Make sure you are in the correct directory (`VisuaLifes/`) before running Python. |
| mkdocs: command not found | Install MkDocs with `pip install mkdocs`. |
| Slow performance during training | Try smaller datasets (e.g., part of CIFAR-10) — VisuaLife is CPU-based. |


##  Summary

| Step | Command |
|------|---------|
| Clone repository | `git clone https://github.com/JeewanAidi/VisuaLifes.git` |
| Create environment | `python -m venv venv` |
| Activate environment | `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux) |
| Install dependencies | `pip install -r requirements.txt` |
| Preview documentation | `mkdocs serve` |


##  Next Steps
Continue with:

- [Getting Started](Getting_Started.md)
- [Architecture Overview](Architecture.md)


© 2025 VisuaLife | Documentation maintained by Jeewan Aidi