import pandas as pandas
import matplotlib.pyplot as pyplot
from matplotlib.backends.backend_pdf import PdfPages

#df = pd.read_csv('telemetry_volvo.csv', index_col=0).iloc[:3000, :]
df = pandas\
        .read_csv('telemetry_1323.csv', index_col=0)\
        .iloc[:3000, :]\
        .interpolate(method='linear', axis=0)
    #.fillna(method="ffill")
    #.interpolate(method='linear', limit_direction='forward', axis=0)

#df.head()

#print()

figure, plots = pyplot.subplots(
    nrows=len(df.columns),
    ncols=1,
    sharex=True,
    figsize=(24, 192))

pyplot.subplots_adjust(hspace=0.5,bottom=0.05,top=0.95)

#

#figure.subplots_adjust(hspace=0.5)

index = 0

#plots[0].plot(df["speed"].values)

for column in df:
    canvas = plots[index]
    canvas.plot(df[column].values)
    canvas.set_xlabel(column)
    index = index + 1

#pyplot.show()

pdf = PdfPages('longplot.pdf')
pdf.savefig()
pdf.close()
