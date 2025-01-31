# Autonomous-Stock-Trading-Using-LSTM-Models
This repository implements an autonomous stock trading application that uses Long Short-Term Memory (LSTM) neural network models to make stock price predictions. Note that the code and results in this repository are for educational and experimental purposes only; always conduct your own research and tests before making real financial decisions.
## Table of Contents
1. [Overview](https://github.com/BurakAhmet/Autonomous-Stock-Trading-Using-LSTM-Models/tree/main?tab=readme-ov-file#overview)
2. [Requirements](https://github.com/BurakAhmet/Autonomous-Stock-Trading-Using-LSTM-Models/tree/main?tab=readme-ov-file#requirements)
3. [Results](https://github.com/BurakAhmet/Autonomous-Stock-Trading-Using-LSTM-Models/tree/main?tab=readme-ov-file#results)
   - [Model Evaluation Results](https://github.com/BurakAhmet/Autonomous-Stock-Trading-Using-LSTM-Models/tree/main?tab=readme-ov-file#model-evaluation-results)
   - [Predictions](https://github.com/BurakAhmet/Autonomous-Stock-Trading-Using-LSTM-Models?tab=readme-ov-file#predictions)
   - [Trading Results](https://github.com/BurakAhmet/Autonomous-Stock-Trading-Using-LSTM-Models?tab=readme-ov-file#trading-results)
5. [Acknowledgements](https://github.com/BurakAhmet/Autonomous-Stock-Trading-Using-LSTM-Models/tree/main?tab=readme-ov-file#acknowledgements)

## Overview
This project aims to:
* Use LSTM-based deep learning models to predict price movements.
* Automate buy/sell decisions based on predicted trends.
The main goal is to explore the use of deep learning in trading strategies.

**For more details you can check the [project report](https://github.com/BurakAhmet/Autonomous-Stock-Trading-Using-LSTM-Models/blob/main/Report.pdf)**

## Requirements
You can download the necessary libraries from the [requirements.txt](https://github.com/BurakAhmet/Autonomous-Stock-Trading-Using-LSTM-Models/blob/main/requirements.txt) with this command:
  ```pip install -r requirements.txt```.

  ## Results
### Model evaluation results
|  Stock Name | MAE  | MSE  | RMSE  |
|---|---|---|---|
|  ASELS |  0.3169 |  0.2575 |  0.5074 |   
| THYAO  |  0.3064 |  0.2622 |  0.5120 |   
| AEFES  | 0.4566  |  0.5370 |  0.7328 |  
|  AFYON | 0.1074  | 0.0239  |  0.1547 |   

### Predictions
**Prediction for AEFES**

![image](https://github.com/user-attachments/assets/2b17ebe8-d013-4069-b8be-9ae0fb4637ef)

**Prediction for AFYON**

![image](https://github.com/user-attachments/assets/9908f7df-1f8f-4852-89ea-51750ff913ba)

**Prediction for ASELS**

![image](https://github.com/user-attachments/assets/6337625e-8214-4c38-9b24-5a0b28ae8939)

**Prediction for THYAO**

![image](https://github.com/user-attachments/assets/1c7ba1b1-9166-4443-b653-d650d90f7ff3)


### Trading Results
**Trading of ASELS**

![image](https://github.com/user-attachments/assets/63f78393-4639-4098-acd8-e61f088868ea)

**Trading of THYAO**

![image](https://github.com/user-attachments/assets/e42a3036-9ea1-41b4-b7ee-c51306e0caea)

**Daily portfolio value while trading of ASELS and THYAO**

![image](https://github.com/user-attachments/assets/6ef69763-ccd8-4402-b76b-d3cb1af23c60)

## Acknowledgements
* Dataset: https://www.kaggle.com/datasets/gokhankesler/borsa-istanbul-turkish-stock-exchange-dataset/data




