# Degradation predictions

### Motion
```
select * from fm_object_service_motion where fmobj_id = 1323
```

### Data export
```
psql -h localhost -p 5432 -U trackpoint -d trackpoint3x -W -f data_export.sql
```

### Useful links
https://habr.com/ru/post/331382/
https://neurohive.io/ru/osnovy-data-science/avtojenkoder-tipy-arhitektur-i-primenenie/
https://ru-keras.com/guide-sequential/

https://keras.io/api/preprocessing/timeseries/

#### Reducing bit depth
https://keras.io/guides/working_with_rnns/
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding


http://www.machinelearning.ru/wiki/index.php?title=%D0%9C%D0%B0%D1%88%D0%B8%D0%BD%D0%BD%D0%BE%D0%B5_%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5_(%D0%BA%D1%83%D1%80%D1%81_%D0%BB%D0%B5%D0%BA%D1%86%D0%B8%D0%B9,_%D0%9A.%D0%92.%D0%92%D0%BE%D1%80%D0%BE%D0%BD%D1%86%D0%BE%D0%B2)

https://timeseries-ru.github.io/course/README.html

https://keras.io/examples/timeseries/timeseries_anomaly_detection/


#### LSTM
https://keras.io/api/layers/recurrent_layers/lstm/


https://www.kaggle.com/sushantjha8/multiple-input-and-single-output-in-keras


#### How to Use the Keras Functional API for Deep Learning
https://machinelearningmastery.com/keras-functional-api-deep-learning/

#### A Quasi-SVM in Keras
https://keras.io/examples/keras_recipes/quasi_svm/

#### Keras implementation of an encoder-decoder for time series prediction using architecture
https://awaywithideas.com/keras-implementation-of-a-sequence-to-sequence-model-for-time-series-prediction-using-an-encoder-decoder-architecture/

https://github.com/guillaume-chevalier/seq2seq-signal-prediction

https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

#### Timeseries anomaly detection using an Autoencoder
https://keras.io/examples/timeseries/timeseries_anomaly_detection/

#### A ten-minute introduction to sequence-to-sequence learning in Keras
https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

#### How to Develop an Encoder-Decoder Model for Sequence-to-Sequence Prediction in Keras
https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/

#### One-to-Many, Many-to-One and Many-to-Many LSTM Examples in Keras
https://wandb.ai/ayush-thakur/dl-question-bank/reports/One-to-Many-Many-to-One-and-Many-to-Many-LSTM-Examples-in-Keras--VmlldzoyMDIzOTM

https://blog.keras.io/building-autoencoders-in-keras.html

https://towardsdatascience.com/lstm-autoencoder-for-extreme-rare-event-classification-in-keras-ce209a224cfb
https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352


#### Priver otdelenija encodera ot dekodera no obuchenie vmeste
https://stackoverflow.com/questions/54928981/split-autoencoder-on-encoder-and-decoder-keras
