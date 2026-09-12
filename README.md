# 🏏 IPL Score Prediction using Deep Learning

A machine learning and deep learning project that predicts the **probable total score of an IPL innings** using historical match data and match-level features such as venue, batting team, bowling team, batsmen, bowler, runs, wickets, and overs.

The project uses **Python, Pandas, Scikit-learn, TensorFlow/Keras, Seaborn, Matplotlib, and ipywidgets** to perform data analysis, preprocessing, neural network training, evaluation, model persistence, and interactive score prediction.

> **Note:** This project was created primarily for **learning and practice purposes** to understand data preprocessing, exploratory data analysis, neural networks, and machine learning model deployment concepts. It is not intended to provide professional or real-world cricket predictions.

---

## 📌 Project Overview

The objective of this project is to build a neural network capable of estimating the **final/total innings score** from the current state and context of an IPL match.

The model learns from historical IPL data and uses features including:

* Venue
* Batting team
* Bowling team
* Batsman/striker
* Bowler
* Current runs
* Wickets
* Overs
* Non-striker

The trained model is then saved and loaded through a separate interactive Jupyter Notebook interface, where users can select match conditions and obtain a predicted total score.

---

## ✨ Key Features

### 📊 Exploratory Data Analysis

The project performs exploratory analysis on the IPL dataset, including:

* Number of matches played at different venues
* Top batsmen based on recorded runs
* Top bowlers based on recorded wickets
* Correlation analysis between numerical features
* Data visualization using Matplotlib and Seaborn

The analysis is implemented in `IPL_Score.py`.

---

### 🔤 Categorical Feature Encoding

Several categorical features are converted into numerical representations using `LabelEncoder`.

The encoded features include:

* `venue`
* `bat_team`
* `bowl_team`
* `batsman`
* `bowler`

The trained encoders are stored in:

```text
label_encoders.joblib
```

This allows the same categorical mappings to be reused when making predictions through the UI.

---

### 📏 Feature Scaling

The numerical feature set is normalized using **MinMaxScaler** before being passed to the neural network.

The trained scaler is saved as:

```text
scaler.joblib
```

The UI subsequently loads this scaler and applies it to user-provided inputs before prediction.

---

### 🧠 Neural Network Model

A fully connected neural network is implemented using **TensorFlow/Keras**.

The architecture consists of:

```text
Input Layer
     │
     ▼
Dense Layer — 512 neurons — ReLU
     │
     ▼
Dense Layer — 216 neurons — ReLU
     │
     ▼
Output Layer — 1 neuron — Linear
```

The model uses:

* **Optimizer:** Adam
* **Loss Function:** Huber Loss
* **Training Epochs:** 10
* **Batch Size:** 64
* **Output:** Predicted total score

The model architecture and training pipeline are implemented in `IPL_Score.py`.

---

### 📈 Model Evaluation

The project evaluates predictions using:

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**

Training and validation loss are also plotted to observe the model's learning behavior.

---

### 🖥️ Interactive Prediction Interface

The `UI.ipynb` notebook provides an interactive interface using **ipywidgets**.

Users can select:

* Venue
* Batting team
* Bowling team
* Striker
* Bowler
* Non-striker

And enter:

* Runs
* Wickets
* Overs

After clicking **Predict Score**, the saved neural network processes the input and displays the predicted total runs.

---

## 🛠️ Technologies Used

| Technology             | Purpose                                             |
| ---------------------- | --------------------------------------------------- |
| **Python**             | Core programming language                           |
| **Pandas**             | Data loading and manipulation                       |
| **NumPy**              | Numerical computation                               |
| **Matplotlib**         | Data visualization                                  |
| **Seaborn**            | Statistical visualization                           |
| **Scikit-learn**       | Encoding, scaling, train/test splitting, evaluation |
| **TensorFlow / Keras** | Neural network development and training             |
| **Joblib**             | Saving/loading encoders and scaler                  |
| **Jupyter Notebook**   | Interactive prediction interface                    |
| **ipywidgets**         | Interactive prediction controls                     |

---

## 📂 Repository Structure

```text
IPL-Score-Prediction/
│
├── IPL_Score.py
│   └── Data analysis, preprocessing,
│       model building, training and evaluation
│
├── UI.ipynb
│   └── Interactive prediction interface
│
├── ipl_data.csv
│   └── Historical IPL match dataset
│
├── model.keras
│   └── Trained TensorFlow/Keras model
│
├── model.h5
│   └── Saved neural network model
│
├── model.joblib
│   └── Serialized model artifact
│
├── label_encoders.joblib
│   └── Saved categorical feature encoders
│
├── scaler.joblib
│   └── Saved MinMaxScaler
│
└── README.md
    └── Project documentation
```

The repository currently contains the dataset, training code, notebook-based UI, and multiple saved model artifacts.

---

## 🔄 Project Workflow

The overall pipeline can be summarized as:

```text
             IPL Historical Dataset
                      │
                      ▼
              Data Loading
                      │
                      ▼
          Exploratory Data Analysis
                      │
                      ▼
          Feature Selection
                      │
                      ▼
        Categorical Label Encoding
                      │
                      ▼
             Train/Test Split
                      │
                      ▼
            Feature Scaling
                      │
                      ▼
          Neural Network Training
                      │
                      ▼
               Model Evaluation
                      │
                      ▼
          Save Model + Encoders
                      │
                      ▼
          Interactive Jupyter UI
                      │
                      ▼
             User Match Inputs
                      │
                      ▼
            Predicted Total Runs
```

---

## 📊 Input Features

The prediction model uses the following features:

| Feature       | Description                             |
| ------------- | --------------------------------------- |
| `venue`       | Stadium/venue where the match is played |
| `bat_team`    | Current batting team                    |
| `bowl_team`   | Current bowling team                    |
| `batsman`     | Current striker                         |
| `bowler`      | Current bowler                          |
| `runs`        | Runs scored so far                      |
| `wickets`     | Wickets lost                            |
| `overs`       | Overs completed                         |
| `non-striker` | Non-striker batsman                     |

The target variable is:

```text
total
```

which represents the innings' total score.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/JoydeepDey-hub/IPL-Score-Prediction.git
```

### 2. Navigate to the Project

```bash
cd IPL-Score-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv ipl-env
```

Activate it on Windows:

```bash
ipl-env\Scripts\activate
```

On Linux/macOS:

```bash
source ipl-env/bin/activate
```

### 4. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib jupyter ipywidgets
```

---

## ▶️ Training the Model

Run the main Python script:

```bash
python IPL_Score.py
```

The script performs:

1. Dataset loading
2. Exploratory data analysis
3. Feature encoding
4. Feature selection
5. Train/test splitting
6. Feature scaling
7. Neural network creation
8. Model training
9. Model evaluation
10. Model saving

The trained Keras model and preprocessing artifacts are saved for later prediction.

---

## 🖥️ Running the Prediction Interface

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
UI.ipynb
```

The notebook loads:

```text
model.keras
label_encoders.joblib
scaler.joblib
```

and creates interactive dropdowns and input fields for making predictions.

After entering the match information, click:

```text
Predict Score
```

The interface displays the estimated total runs.

---

## 🧪 Example Prediction Workflow

A typical prediction follows this process:

```text
Select Venue
      ↓
Select Batting Team
      ↓
Select Bowling Team
      ↓
Select Striker
      ↓
Select Bowler
      ↓
Enter Current Runs
      ↓
Enter Wickets
      ↓
Enter Overs
      ↓
Select Non-Striker
      ↓
       Predict Score
            ↓
      Neural Network
            ↓
    Predicted Total Runs
```

---

## 💾 Saved Model Artifacts

The repository contains multiple saved model/preprocessing artifacts:

### `model.keras`

The TensorFlow/Keras model used by the interactive prediction notebook.

### `model.h5`

An additional saved representation of the neural network model.

### `model.joblib`

A serialized model artifact stored using Joblib.

### `label_encoders.joblib`

Contains the fitted `LabelEncoder` objects used for categorical features.

### `scaler.joblib`

Contains the fitted `MinMaxScaler` used to normalize model inputs.

The UI specifically loads `model.keras`, `label_encoders.joblib`, and `scaler.joblib`.

---

## 📈 Evaluation Metrics

The project calculates:

### Mean Absolute Error — MAE

Measures the average absolute difference between the predicted and actual scores.

```text
MAE = Average |Actual - Predicted|
```

### Mean Squared Error — MSE

Measures the average squared difference between predictions and actual values.

```text
MSE = Average (Actual - Predicted)²
```

Both metrics are calculated after model prediction in the training pipeline.

---

## 🎯 Project Objectives

The main objectives of this project were to:

* Understand a real-world regression problem.
* Work with historical IPL data.
* Perform exploratory data analysis.
* Prepare categorical and numerical features.
* Apply feature encoding and scaling.
* Build a neural network using TensorFlow/Keras.
* Train a model for score prediction.
* Evaluate regression performance.
* Save trained models and preprocessing objects.
* Build an interactive prediction interface using Jupyter widgets.

---

## 📚 Learning Outcomes

This project helped develop practical understanding of:

* Data preprocessing
* Exploratory Data Analysis
* Feature engineering
* Categorical encoding
* Feature scaling
* Train/test splitting
* Neural network architecture
* Regression using deep learning
* TensorFlow/Keras
* Model evaluation
* Model serialization
* Interactive ML interfaces
* Using trained models for inference

---

## ⚠️ Limitations

This project is primarily a **learning and practice project**, so the predictions should not be interpreted as professional cricket forecasts.

Current limitations include:

* The model is trained on historical data.
* Cricket conditions can change significantly between matches.
* Player form and real-time performance are not fully represented.
* Weather conditions are not included.
* Pitch condition information is not explicitly modeled.
* Toss information is not directly incorporated.
* Live ball-by-ball data is not connected.
* The prediction interface is notebook-based rather than a deployed web application.
* Model performance depends heavily on the quality and distribution of the historical dataset.

---

## 🔮 Future Improvements

Potential improvements include:

* [ ] Add more recent IPL seasons to the training data
* [ ] Incorporate real-time match data
* [ ] Add weather information
* [ ] Include pitch conditions
* [ ] Include toss winner and toss decision
* [ ] Add player-form statistics
* [ ] Add recent team performance
* [ ] Experiment with Random Forest, XGBoost and other regression models
* [ ] Compare multiple neural network architectures
* [ ] Perform systematic hyperparameter tuning
* [ ] Add cross-validation
* [ ] Improve feature engineering
* [ ] Build a Flask/FastAPI backend
* [ ] Create a proper HTML/CSS/JavaScript frontend
* [ ] Deploy the prediction application online

---

## 🧠 Model Architecture

The current neural network uses a simple feed-forward architecture:

```text
9 Input Features
       │
       ▼
Dense(512, ReLU)
       │
       ▼
Dense(216, ReLU)
       │
       ▼
Dense(1, Linear)
       │
       ▼
Predicted Total Score
```

The model is trained using the **Adam optimizer** with **Huber Loss**, which provides a regression loss that is less sensitive to large errors than standard squared-error loss.

---

## 🎓 Educational Purpose

This project was created as a **practice/learning project** while exploring machine learning and deep learning concepts.

It demonstrates the complete basic workflow of an ML project:

```text
Data
  ↓
Analysis
  ↓
Preprocessing
  ↓
Feature Engineering
  ↓
Model Development
  ↓
Training
  ↓
Evaluation
  ↓
Model Saving
  ↓
Inference
```

The primary goal is to demonstrate practical implementation rather than to provide production-grade cricket analytics.

---

## 👨‍💻 Author

**Joydeep Dey**

B.Tech — Computer Science & Engineering
Artificial Intelligence & Machine Learning

GitHub:
https://github.com/JoydeepDey-hub

---

## 📄 Disclaimer

This project is intended **strictly for educational and experimental purposes**.

The predicted IPL scores are machine learning estimates and should not be considered guaranteed or official predictions.

The project is not affiliated with, sponsored by, or officially connected to the **Indian Premier League (IPL)**, BCCI, or any IPL franchise.

---

## ⭐ If You Find This Project Useful

If this project helps you understand machine learning, deep learning, or sports analytics, consider giving the repository a ⭐.

**Built for learning. Built for experimentation. Built with Python & Deep Learning. 🏏🤖**
