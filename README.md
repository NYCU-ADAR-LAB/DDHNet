# DDHNet
A Dense Data Honoring Nerual Network Model for Click-Through Rate Predicton 
# Model Architecture
![Uploading image.png…]()

## key technology
1. dense data preprocessing (percentile clamping & mirror normalization)
2. bidirectional fusion (D2S MLP & S2D MLP)
3. multi-expert bottom MLPs
4. front gate of expert

# Model Configuration 
1. percentile clamping (99.25%)
2. 2 experts
   
# Requirements
We tested DDHNet with the following requirements.
python => 3.8.10
pytorch => 1.10

# Results
| Year | Model | AUC | FLOPs | 
| ---- | ----- | --- | ----- |
| 2017 | DeepFM | 0.8007 | 967K |
# Reproduce the results 
1. download CTR kaggle dataset => http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/
2. percentile clamping dataset
3. mirror normalization
4. train model
