from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler

DATA_FILE = 'telemetry_volvo.csv'

def series_to_supervised(data):
    n_vars = 1 if type(data) is list else data.shape[1]

    #df = DataFrame(data)
    cols, names = list(), list()

    cols.append(data.shift(1))
    names += [('var%d(t-%d)' % (j + 1, 1)) for j in range(n_vars)]

    cols.append(data.shift(-0)[0])
    names += ['var1(1)']

    agg = concat(cols, axis=1)
    agg.columns = names

    return agg

# load dataset
df = read_csv(DATA_FILE, index_col=0) \
        .iloc[:60, :] \
        .interpolate(method='linear', axis=0)\
        .drop(columns=['total_odometer', 'gsm_signal', 'sped'])

values = df.values
#print(df.head())
#print(df.tail())

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = DataFrame(scaler.fit_transform(values))

#Tut nam nuzno vzjat vse kolonki i obedenit v odnu, shto bi poluchit kompozitnij znachenie
#print(scaled.head())

#reframed = series_to_supervised(scaled)

#Udaljaem NaN
#reframed.dropna(inplace=True)


#print("supervised series")
#print(reframed.head())

test_size = 10
test = scaled.iloc[:test_size, :]

#test_x, test_y = test.values[:, :-1], test.values[:, -1]

print()
