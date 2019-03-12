import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics

#Initialize data frame
df = pd.read_csv("Admission_Predict_Ver1.1.csv")

#Visualizing Data at hand
x = df["GRE Score"] #adjust X here
print("Mean: "+str(np.mean(x)))
print("Stdev: "+str(statistics.stdev(x)))

plt.scatter(x, df.iloc[:,-1])
plt.show()

plt.hist(x)
plt.show()

#Creating LSRL(Regression Line) for GRE Scores
x = df["GRE Score"]
y = df.iloc[:,-1]

fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit)

r = np.corrcoef(x,y)

plt.plot(x,y, 'yo', x, fit_fn(x), '--k')
plt.title("GRE vs Chance Admit (r="+str(r[0][1])+")", fontsize=18)
plt.xlabel('GRE Scores')
plt.ylabel('Chance of Admit')
plt.show()

#Residuals Plot for GRE vs Chance Admit
resid = []
for i in range(len(y)):
	yhat = 0.006883*x[i] - 1.368 #adjust yhat here
	resid_val = y[i] - yhat
	resid.append(resid_val)
	
r = np.corrcoef(x,resid)
print("r="+str(r[0][1]))

plt.scatter(x,resid)
plt.title("Residuals for GRE vs Chance Admit")
plt.show()

#Seperating out by University Rank

ranknum = 1 #adjust rank level here
xvar = "CGPA" #adjust x var here

rank = []
for i in range(len(df["University Rating"])):
	if df["University Rating"][i] == ranknum:
		rank.append(i)
x=[]
y=[]
for i in range(len(rank)):
	x.append(df[xvar][rank[i]])
for i in range(len(rank)):
	y.append(df.iloc[:,-1][rank[i]])

print("Mean: "+str(np.mean(x)))
print("Std: "+str(statistics.stdev(x)))
print("n: "+str(len(x)))

r = np.corrcoef(x,y)
print("r="+str(r[0][1]))

fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit)

plt.plot(x,y, 'yo', x, fit_fn(x), '--k')
plt.title("Rank "+str(ranknum), fontsize=18)
plt.xlabel(xvar)
plt.ylabel('Chance of Admit')
plt.show()

#Confidence Intervals were calculated on a TI-84 calculator