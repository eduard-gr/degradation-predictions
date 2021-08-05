import pandas as pandas
import matplotlib.pyplot as pyplot
from matplotlib.backends.backend_pdf import PdfPages
from pandas import read_csv, DataFrame
from scipy.stats import stats

DATA_FILE = 'telemetry_volvo.csv'

df = read_csv(DATA_FILE, index_col=0) \
        .iloc[:3000, :] \
        .drop(columns=[
            'total_odometer',
            'gsm_signal',
            'number_of_dtc',
            'vehicle_speed',
            'runtime_since_engine_start',
            'distance_traveled_mil_on',
            'fuel_level',
            'distance_since_codes_clear',
            'absolute_load_value',
            'time_since_codes_cleared',
            'absolute_fuel_rail_pressure',
            'engine_oil_temperature',
            'fuel_injection_timing',
            'fuel_rate',
            'battery_current',
            'gnss_status',
            'data_mode',
            'gnss_pdop',
            'gnss_hdop',
            'sleep_mode',
            'ignition',
            'movement',
            'active_gsm_operator',
            'green_driving_type',
            'unplug',
            'green_driving_value',
            'sped']) \
        .interpolate(method='linear', axis=0) \
        .fillna(0)


#source_values = raw.values
#source_values = source_values.astype('float32')

#Primenit metod zscoringa https://www.statology.org/z-score-python/
source_values = stats.zscore(df.values, axis=1)

df = DataFrame(
    source_values,
    columns=df.columns,
    index=df.index)

figure, plots = pyplot.subplots(
    nrows=df.shape[1],
    ncols=1,
    sharex=True,
    figsize=(24, 192))

pyplot.subplots_adjust(hspace=0.5,bottom=0.05,top=0.95)
index = 0

for column in df:
    canvas = plots[index]
    canvas.plot(df[column].values)
    canvas.set_xlabel(column)
    index = index + 1

#pyplot.show()

pdf = PdfPages('longplot.pdf')
pdf.savefig()
pdf.close()
