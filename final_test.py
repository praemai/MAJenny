import seaborn as sns
import csv
import pandas as pd
import matplotlib.pyplot as plt

#data =  {}
#with open('/home/mbiadmin/Desktop/ErgebnisseClassification/Ergebnisse_Classifcation_Reference_ROI.csv') as csvfile:
##   results = csv.DictReader(csvfile, delimiter =',')

    #data[row].Category = row[0]
filename= '/home/mbiadmin/Desktop/ErgebnisseClassification/Ergebnisse_Classifcation_Reference_ROI.csv'
to_plot = pd.read_csv(filename, sep=',', header='infer')

#fig, ax = plt.subplots(figsize=(4,4))

plot =sns.swarmplot(x="Category", y="AUC", data=to_plot,size =8, hue="Label", palette="colorblind")
#plot =sns.stripplot(x="Category", y="AUC", data=to_plot,size=4, jitter=0.01)
#plot = sns.factorplot(x="Category", y="AUC", hue="Label",
               #col="Category", data=to_plot, kind="swarm");

#ax1 = plt.axes([0, 0.6, 3, 0.5])

plot.set(xlabel= "# Images")
#ax = plot.add_subplot(111)h_pad
#plot.annotate('annotate', xy=(2, 1), xytext=(3, 4),
 #           arrowprops=dict(facecolor='black', shrink=0.05))
#plot.text(0.95, 0.01, 'colored text in axes coords',
#        verticalalignment='bottom', horizontalalignment='right',
#        transform=plot.transAxes,
#        color='green', fontsize=15)
#for line in range(0,to_plot.shape[0]):
#    if line%2 ==0:
#        plot.text(to_plot.Cat[line], to_plot.AUC[line], to_plot.Label[line], size=5,horizontalalignment='left')
#    else:
#        plot.text(to_plot.Cat[line], to_plot.AUC[line], to_plot.Label[line], size=5, horizontalalignment='right')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#ax = plot.get_yaxis()
#plot.annotate("Saturday\nMean",
#            xy=(2, "Category"), xycoords='data',
#            xytext=(.5, .5), textcoords='axes fraction',
#            horizontalalignment="center",
#            arrowprops=dict(arrowstyle="->",
 #                           connectionstyle="arc3"),
#            bbox=dict(boxstyle="round", fc="w"),)
plt.show(plot)
#
# y= to_plot['Category']
# z= to_plot['AUC']
# n= to_plot['Label']
#
# fig, ax = plt.subplot()
# ax.scatter =()
#
# for i, txt in enumerate(n):
#     ax.annotate(txt, (z[i],y[i]))