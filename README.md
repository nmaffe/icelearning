# Artificial intelligence techniques for ice core analyses 

The ICELEARNING project ..

### Create training dataset 🏋️ & train the model 🤖
---
```
python model_train.py
```

### Model inference on GRIP ice core samples 🕵🏿
---
```
python model_test.py
```
This code consists of the following steps:
* The trained model .pth is loaded
* The GRIP dataset is loaded (3M images)
* Inference loop
* Final dataset saved: ```dataset/test/inference_on_GRIP_samples.csv```

## Acknowledgments
---
[<img aligh="right" alt="EU" src="img/logo_MSCA.png" height="70" />](https://marie-sklodowska-curie-actions.ec.europa.eu/)

The ICELEARNING project is supported by the European Union’s Horizon 2020 Marie Skłodowska-Curie Actions (grant no. 845115).
