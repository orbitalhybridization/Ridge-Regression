from sys import argv
import pandas as pd
import numpy as np
import pdb
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_validate, RepeatedKFold
from sklearn.linear_model import LinearRegression,Ridge
import seaborn as sb

ATTRIBUTE_LABELS = {'symboling':0,'normalized-losses':1,'make':2,'fuel-type':3,
					'aspiration':4,'num-of-door':5,'body-style':6,'drive-wheels':7,
					'engine-location':8,'wheel-base':9,'length':10,'width':11,'height':12,
					'curb-weight':13,'engine-type':14,'num-of-cylinders':15,'engine-size':16,
					'fuel-system':17,'bore':18,'stroke':19,'compression-ratio':20,'horsepower':21,
					'peak-rpm':22,'city-mpg':23,'highway-mpg':24,'price':25}

LABELS_OF_INTEREST = ['wheel-base','length','width','height','curb-weight','engine-size',
					'bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg',
					'highway-mpg']

TARGETS = ['price']

def main(filename,display=True):

	print("Running main function")
	print("---------------------------")

	print("Importing Data")
	print("---------------------------")
	data = import_data(filename,unknown_char='?')

	print("Preprocessing")
	print("---------------------------")
	data = preprocess(data)

	print("Plotting Initial Data")
	print("---------------------------")
	plot_vs_target(data,LABELS_OF_INTEREST,TARGETS,subplot_dim=[3,5]) # make a 3x5 subplot

	print("Plotting Pairwise Data")
	print("---------------------------")
	plot_pairwise(data,LABELS_OF_INTEREST,subplot_dim=[4,4])
	
	print("Computing Model 1")
	print("---------------------------")
	model1 = linear_model(data,['wheel-base','engine-size'],'price')

	print("Computing Model 2")
	print("---------------------------")
	data['highway-mpg-inverse'] = 1/data['highway-mpg'] # invert highway-mpg
	model2 = linear_model(data,['wheel-base','curb-weight','highway-mpg-inverse'],'price')

	print("Computing Model 3")
	print("---------------------------")
	model3 = linear_model(data,['length','height','horsepower'],'price')

	print("Performing Cross-Validation on Model 1")
	print("---------------------------\n")
	cross_validation(data,['wheel-base','engine-size'],'price',regression_type='ridge')
	print("\n\n")

	print("Performing Cross-Validation on Model 2")
	print("---------------------------\n")
	cross_validation(data,['wheel-base','curb-weight','highway-mpg-inverse'],'price',regression_type='ridge')
	print("\n\n")

	print("Performing Cross-Validation on Model 3")
	print("---------------------------\n")
	cross_validation(data,['length','height','horsepower'],'price',regression_type='ridge')
	print("\n\n")

	
	if display: display_plots()

	print("Done.")

def import_data(filename,unknown_char=np.nan):
	df = pd.read_csv(filename,names=ATTRIBUTE_LABELS.keys())
	df = df.replace(unknown_char,np.nan) # replace unknowns with column means
	return df

def preprocess(df):

	#df = df.dropna() # remove empty rows

	# remove features not of interest
	df = df[LABELS_OF_INTEREST+TARGETS]

	df = df.astype(float) # make float

	# remove data points for which target variable is unknown
	for label in LABELS_OF_INTEREST+TARGETS: # go through columns	
		df.fillna(df[label].mean(skipna=True),inplace=True)

	return df


def plot_vs_target(df,variables,targets,subplot_dim=[1,1],print_figures=False):
	index=1
	if print_figures: plt.figure(1)
	for var in variables:
		for target in targets:
			#pdb.set_trace()
			#df.plot.scatter(x=var,y=target)
			if print_figures:
				plt.subplot(subplot_dim[0],subplot_dim[1],index,title=var+' vs. '+target)
				plt.scatter(list(df[var]),list(df[target]))
			index+=1

def plot_pairwise(df,variables,subplot_dim=[1,1],print_figures=False):
	index = 1
	if print_figures: plt.figure(2)
	for var1 in variables: # get every third
		for var2 in ['compression-ratio']:
			if var1 == var2:
				continue
			if print_figures:
				plt.subplot(subplot_dim[0],subplot_dim[1],index,title=var1+' vs. '+var2)
				plt.scatter(list(df[var1]),list(df[var2]))
				plt.xlabel(var1)
				plt.ylabel(var2)
			index+=1

def linear_model(df,variables,target,print_figures=False,print_results=False):

	#perform any necessary transformations on the data

	# split data
	#x_train,x_test,y_train,y_test = train_test_split(df[variables],df[target],test_size=0.25)

	# perform linear regression
	model = LinearRegression()
	model = model.fit(df[variables],df[target])
	predicted_target = model.predict(df[variables])
	if print_figures:
		plt.figure()
		plt.scatter(df[target],predicted_target)
		plt.xlabel("Actual "+target)
		plt.ylabel("Predicted "+target)
	if print_results:
		# extract weights
		print("Model parameters:")
		print(model.coef_)
		print()
		print("R^2 for this model:")
		print(model.score(df[variables],df[target]))
		print("\n-----------------------------------------------------\n")
	return model

def cross_validation(df,variables,target,regression_type='linear',reps=3,folds=10,print_figures=True,print_results=True):
	
	# check regression type
	if regression_type == 'linear':
		model = LinearRegression()
	elif regression_type == 'ridge':
		model = Ridge(alpha=2.0)
	else:
		raise ValueError("Yooo Invalid regression type chosen!")

	# KFolds
	rkf = RepeatedKFold(n_splits=folds,n_repeats=reps) # get a kfold generator
	cv = cross_validate(model,df[variables],df[target],cv=rkf,scoring=('r2','neg_mean_squared_error'),return_estimator=True)

	MSEs = [error*-1 for error in cv['test_neg_mean_squared_error']] # pull mses

	# calculate expected mse
	expected_mse = np.mean(MSEs)

	# plot kernel density from MSE
	if print_figures:
		sb.kdeplot(data=MSEs).set(title="KDE plot vs. E(MSE)="+str(format(expected_mse,'.1E')))
		plt.show()

	# compute variance
	fitted_models = cv['estimator'] # pull fitted models - 30
	
	# avg prediction across models
	L = reps*folds # num models
	N = np.shape(df[variables])[0] # num data points

	avg_predictions = []
	for n in range(N):
		prediction_sum = 0
		for l in range(L):
			current_model = fitted_models[l]
			current_model_prediction = current_model.intercept_
			for v in range(len(variables)):
				current_model_prediction += df[variables[v]].iloc[n]*current_model.coef_[v] # make prediction (sklearn predict did not like this)
			prediction_sum += current_model_prediction
		prediction_sum /= L # avg over models
		avg_predictions.append(prediction_sum) # save to averages and move to next obs

	# calculate variance	
	variance = 0
	for n in range(N):
		for l in range(L):
			yl_xn = fitted_models[l].intercept_ # predicted price
			for v in range(len(variables)):
				yl_xn += df[variables[v]].iloc[n]*fitted_models[l].coef_[v] # make prediction (sklearn predict did not like this)
			ybar_xn = avg_predictions[n] # avg prediction
			variance += ((yl_xn - ybar_xn)**2)

	variance /= (L*N) # divide by L*N

	# get bias^2 + noise
	# pretty sure we can just do:
	bias_sq_noise = expected_mse - variance


	# Print Results
	if print_results:
		print("Expected MSE: " + format(expected_mse,'.1E'))
		print("Variance: " + str(variance))
		print("Bias^2 + Noise: " + str(bias_sq_noise))

def display_plots():
	plt.show()


if __name__ == "__main__":

	main(argv[1])