from pandas import read_csv, DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as pyplot
from tensorflow.python.saved_model import saved_model


DATA_FILE = 'sample_telemetry_volvo.csv'

#parse_dates=True,
raw = read_csv(DATA_FILE, index_col=0) \
        .interpolate(method='linear', axis=0)\
        .drop(columns=['total_odometer', 'gsm_signal', 'sped'])\
        .fillna(0)

source_values = raw.values
source_values = source_values.astype('float32')

source_values = MinMaxScaler(feature_range=(-1, 1)).fit_transform(source_values)

encoder = saved_model.load('volvo_encoded_model_tanh')
decoder = saved_model.load('volvo_decoder_model_tanh')

result_df = DataFrame(
    encoder.signatures["predict"](source_values),
    columns=['predict'],
    index=raw.index)


pyplot.figure()
pyplot.subplots_adjust(hspace=0.5,bottom=0.05,top=0.95)
pyplot.subplot(1, 1, 1)
pyplot.plot(result_df.values)
pyplot.ylabel('predict')
pyplot.show()