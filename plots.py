import seaborn as sns
import matplotlib
import pandas as pd

sns.set_theme()
df = pd.read_csv('data/heating.csv', usecols=[0,1,3], nrows=11)

sns.relplot(
    data=df, kind="line",
    x="time", y="temp",
    palette='rocket', 
    #hue='state'style="state",markers=True
)

matplotlib.pyplot.show() 