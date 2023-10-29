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
| Year | Model | FLOPs | AUC | 
| ---- | ----- | ----- | --- |
| 2017 | DeepFM | 967K | 0.8007 |
| 2017 | DCN | 4.2M | 0.8026 |
| 2018 | xDeepFM | 68.7M | 0.8052 |
| 2019 | AutoInt+ | 4.3M | 0.8061 |
| 2019 | DLRM | 960K | 0.8114 |
| 2020 | Deeplight | 970K | 0.8123 |
| 2020 | DCN v2 | 6.3M | 0.8115 |
| 2021 | MaskNet | 4.6M | 0.8131 |
| 2023 | FinalMLP | 4.9M | 0.8149 |
| 2023 | DDHNet | 3.3M | 0.8159 |

# Reproduce the results 
1. download CTR kaggle dataset => http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/
2. percentile clamping dataset
3. mirror normalization
4. train model
