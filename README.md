# Development of an ANN for Binary Classification Using the `make_moons` Dataset

## Project Overview
This project builds an Artificial Neural Network (ANN) with TensorFlow/Keras to classify the two interleaving half-moons produced by Scikit-learn's `make_moons()`. The data is nonlinear, so a straight-line classifier struggles, which makes it ideal for showing what a neural network can do. The ANN is compared against a Logistic Regression baseline.

## Objectives
- Understand binary classification using an ANN
- Generate, preprocess and visualize a nonlinear dataset
- Build, train and evaluate an ANN (accuracy, precision, recall, F1, confusion matrix)
- Visualize the learned decision boundary
- Predict new inputs and compare with Logistic Regression

## Technologies Used
Python 3.9 to 3.12, NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow/Keras, Jupyter Notebook / Google Colab

## Project Structure
```
ann-make-moons/
├── ann_make_moons.ipynb   # step-by-step notebook with explanations
├── main.py                # standalone script (core workflow)
├── requirements.txt
└── README.md
```

## Model Summary
| Layer | Details |
|---|---|
| Input | 2 features |
| Hidden 1 | 16 neurons, ReLU |
| Hidden 2 | 8 neurons, ReLU |
| Output | 1 neuron, sigmoid |

Optimizer: Adam | Loss: binary cross-entropy | Batch size: 32 | Max epochs: 100 | EarlyStopping (patience 10, best weights restored)

## Installation and Execution

### Windows (Command Prompt or PowerShell)
```bat
cd ann-make-moons
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Run the script:
```bat
python main.py
```
Open the notebook:
```bat
jupyter notebook ann_make_moons.ipynb
```
Then choose **Run > Run All Cells** (or Kernel > Restart & Run All).

If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Google Colab (no installation needed)
1. Go to https://colab.research.google.com and choose **File > Upload notebook**.
2. Upload `ann_make_moons.ipynb`.
3. Choose **Runtime > Run all**. TensorFlow, Scikit-learn and the other libraries are pre-installed.

To run `main.py` in Colab, upload it using the Files panel, then in a cell run:
```python
!python main.py
```

## Expected Outputs
- First rows of the dataset, number of samples (1000) and features (2)
- Scatter plot of the two classes
- Model summary and training logs
- Accuracy and loss curves (training vs validation)
- Test loss, accuracy, precision, recall, F1-score, confusion matrix, classification report
- ANN decision boundary plot with a curved boundary
- Predictions for new sample points (class + probability)
- Comparison table and bar chart: ANN vs Logistic Regression

`main.py` also saves all plots to an `outputs/` folder. Exact numbers vary slightly between machines and TensorFlow versions, but on a typical run the ANN scores around 0.97 to 1.00 accuracy and clearly outperforms Logistic Regression (around 0.85 to 0.92).

## Common Errors and Fixes
| Problem | Fix |
|---|---|
| `No matching distribution found for tensorflow` | TensorFlow needs 64-bit Python 3.9 to 3.12. Check with `python --version`, install a supported version and recreate the virtual environment. |
| `ModuleNotFoundError: No module named 'tensorflow'` | Activate the virtual environment, then `pip install -r requirements.txt`. In Jupyter, select the kernel from that environment. |
| `DLL load failed` (Windows) | Install the Microsoft Visual C++ Redistributable (2015-2022, x64), then reinstall TensorFlow. |
| `'jupyter' is not recognized` | Run `python -m notebook` instead. |
| `oneDNN custom operations` / CUDA warnings | Harmless information messages. To hide them: `set TF_ENABLE_ONEDNN_OPTS=0`. |
| Plot windows do not appear | Set `SHOW_PLOTS = False` in `main.py` and open the PNG files in `outputs/`. |
| Still cannot install TensorFlow | Use the Google Colab option above. |
