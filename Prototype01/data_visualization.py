import pandas as pandas
import matplotlib.pyplot as pyplot
from matplotlib.backends.backend_pdf import PdfPages
from pandas import read_csv, DataFrame
from scipy.stats import stats

from common import load_data_set

DATA_FILE = 'telemetry_volvo.csv'

source = load_data_set(DATA_FILE, 3000)

#source_values = raw.values
#source_values = source_values.astype('float32')

#Primenit metod zscoringa https://www.statology.org/z-score-python/
source_values = stats.zscore(source.values, axis=1)

df = DataFrame(
    source_values,
    columns=source.columns,
    index=source.index)


figure, plots = pyplot.subplots(
    nrows=17,
    ncols=1,
    sharex=True,
    figsize=(24, 192))

pyplot.subplots_adjust(hspace=0.1,bottom=0.05,top=0.25)

index=0

plots[index].plot(df['engine_load'].values, label='engine_load')
plots[index].legend()
index += 1

plots[index].plot(df['speed'].values, label='speed')
plots[index].legend()
index += 1

plots[index].plot(df['engine_rpm'].values, label='engine_rpm')
plots[index].legend()
index += 1

plots[index].plot(df['throttle_position'].values, label='throttle_position')
plots[index].legend()
index += 1


plots[index].plot(df['engine_load'].values, label='engine_load')
plots[index].plot(df['speed'].values, label='speed')
plots[index].plot(df['engine_rpm'].values, label='engine_rpm')
plots[index].plot(df['throttle_position'].values, label='throttle_position')
plots[index].legend()
index += 1




plots[index].plot(df['coolant_temperature'].values, label='coolant_temperature')
plots[index].legend()
index += 1

plots[index].plot(df['intake_air_temperature'].values, label='intake_air_temperature')
plots[index].legend()
index += 1

plots[index].plot(df['ambient_air_temperature'].values, label='ambient_air_temperature')
plots[index].legend()
index += 1

plots[index].plot(df['intake_map'].values, label='intake_map')
plots[index].legend()
index += 1

plots[index].plot(df['barometic_pressure'].values, label='barometic_pressure')
plots[index].legend()
index += 1

plots[index].plot(df['egr_error'].values, label='egr_error')
plots[index].legend()
index += 1

plots[index].plot(df['coolant_temperature'].values, label='coolant_temperature')
plots[index].plot(df['intake_air_temperature'].values, label='intake_air_temperature')
plots[index].plot(df['ambient_air_temperature'].values, label='ambient_air_temperature')
plots[index].plot(df['intake_map'].values, label='intake_map')
plots[index].plot(df['barometic_pressure'].values, label='barometic_pressure')
plots[index].plot(df['egr_error'].values, label='egr_error')
plots[index].legend()
index += 1


plots[index].plot(df['control_module_voltage'].values, label='control_module_voltage')
plots[index].legend()
index += 1

plots[index].plot(df['external_voltage'].values, label='external_voltage')
plots[index].legend()
index += 1

plots[index].plot(df['battery_voltage'].values, label='battery_voltage')
plots[index].legend()
index += 1

plots[index].plot(df['control_module_voltage'].values, label='control_module_voltage')
plots[index].plot(df['external_voltage'].values, label='external_voltage')
plots[index].plot(df['battery_voltage'].values, label='battery_voltage')
plots[index].legend()
index += 1

plots[index].plot(df['maf'].values, label='maf')
plots[index].legend()
index += 1


pdf = PdfPages('longplot.pdf')
pdf.savefig()
pdf.close()

#pyplot.show()
exit(0);

index = 0

for column in df:
    canvas = plots[index]
    canvas.plot(df[column].values, label=column)
    canvas.legend()
    index = index + 1

#pyplot.show()

pdf = PdfPages('longplot.pdf')
pdf.savefig()
pdf.close()
