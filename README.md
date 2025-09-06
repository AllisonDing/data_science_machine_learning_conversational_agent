# Conversational ML Agent

A conversational data science and machine learning agent powered by NVIDIA GPUs.  
You can interact with it using natural language to run data exploration and machine learning tasks with minimal setup.

---

## 📊 Dataset

This project uses the Kaggle [Titanic-Dataset.csv](https://www.kaggle.com/datasets/yasserh/titanic-dataset?select=Titanic-Dataset.csv).

Download the dataset and place it in the `data/` directory before running the agent.

For performance testing, you can also use **Titanic-Dataset-1.csv**, an extrapolated version scaled to 1M rows. This larger dataset is useful for demonstrating GPU acceleration.

---

## Running the Agent

You can run the agent in two different modes:

### 1. **GPU-Accelerated Mode** (NVIDIA cuML + cuDF)
Leverages NVIDIA's RAPIDS libraries for faster data processing and model training.

```bash 
python -m cudf.padnas -m cuml.accel run_chat.py
```

### 2. **CPU Mode** (scikit-learn + pandas)
Uses standard pandas and scikit-learn for data processing and modeling.

```bash
python run_chat.py
```

### The agent supports queries such as:<br>
   describe the data<br>
   preview the head<br>
   train the classification model<br>
   hyperparameter optimization (HPO)<br>
   best practice to perform machine learning<br>
   ...

---

**Note:**  
- Ensure you have the appropriate dependencies installed for each mode.  
- GPU mode requires a supported NVIDIA GPU and the RAPIDS ecosystem installed.
