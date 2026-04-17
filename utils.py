
#######################################################
# Title: Batch Adaptive Causal Annotations
# Last updated: May 21, 2025
######################################################


import os
import sys
sys.path.insert(1, '../')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import pandas as pd
import cvxpy as cp
import xgboost as xgb
import scipy
from scipy.stats import norm, bernoulli, uniform
from scipy.special import expit, logit
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, GammaRegressor, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from scipy.optimize import nnls
from matplotlib.ticker import FormatStrFormatter, LogLocator, ScalarFormatter, NullFormatter
from econml.grf._base_grf import BaseGRF
from econml.utilities import cross_product
import scipy.stats
import statsmodels.api as sm
from forestriesz import ForestRieszATE


def aipw_estimator(Z=None, R=None, Y=None, mu1=None, mu0=None, e1_x=None, pi=None, prop_score='plug-in'):
	"""
	Computes AIPW and variance estimates.

	Inputs: 
		- Z: treatment vector 
		- R: expert label = 1, missingness =0
		- Y: ground truth outcome 
		- mu1, mu0: outcome model vectors of mu(X)
		- e1, e0: propensity scores 
		- pi1, pi0: sampling probabilities
		- prop_score: 'plug-in' for ZR estimate or 'balance' for balancing weights

	Outputs: 
		AIPW, Variance of AIPW
	""" 
	## TODO: flatten dimensions of data 
	if prop_score == 'plug-in':
		# for Z=1
		y1 = Y[Z==1]
		r1 = R[Z==1]
		scores1 = e1_x[Z==1] 
		probs1 = pi[Z==1]
		phi_1 = mu1 + (y1 - mu1)*r1/(scores1 * probs1)

		# for Z=0 
		y0 = Y[Z==0]
		r0 = R[Z==0]
		scores0 = 1-e1_x[Z==0] 
		probs0 = pi[Z==0]
		phi_0 = mu0 + (y0 - mu0)*r0/(scores0 * probs0)

	elif prop_score == 'balance': 
		# for Z=1
		y1 = Y[Z==1]
		r1 = R[Z==1]
		phi_1 = mu1 + (y1 - mu1)*r1*e1_x[Z==1]

		# for Z=0 
		y0 = Y[Z==0]
		r0 = R[Z==0]
		phi_0 = mu0 + (y0 - mu0)*r0*e1_x[Z==0]

	ate = np.mean(phi_1) - np.mean(phi_0)
	var_ATE = (np.var(phi_1) / sum(Z==1)) + (np.var(phi_0) / sum(Z==0))

	return ate, var_ATE


def sampling_r(pi_vals=None,n=None, budget=None): 
	'''
	Randomly selects which datapoints to label according to budget or given pi values.

	Inputs: 
		- pi_vals: either random sampled or from optimization from get_pi() 
		- n: total number of datapoints 
		- budget: annotation budget
	Outputs: 
		- R: indicator for expert label or not

	'''
	if pi_vals is None: 
		R = bernoulli.rvs([budget]*n)
	else:
		R = bernoulli.rvs(pi_vals)
	return R

def predict_func(X=None,L=None,Y=None,pred_cols_list=None,model=None,model_type='nnls'):
	'''
	Predictions for given data points from pretrained model.

	Inputs: 
		- X: numpy array covariates 
		- L: numpy array llm predictions 
		- Y: numpy array outcomes
		- pred_cols_list: names of llm predictions columns 
		- model: prediction model (i.e. outcome model)
		- model_type: type of model (i.e. xgboost, nnls (linear model), llm-2 (uses random forest model)) 
	Outputs:
		- model predictions
	'''
	if model_type=='xgb':
		preds=np.clip(model.predict(xgb.DMatrix(X)), 0, np.inf)
	elif model_type == 'nnls':
		X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
		preds=np.clip(X_with_intercept @ model,0,np.inf)
	elif model_type == 'llm-2':
		rf = RandomForestRegressor(max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100,random_state=42)
		model_mu = rf.fit(X,Y)
		mu_x = model_mu.predict(X)
		llm_train = random_sample_predictions(L,pred_cols_list)
		X_df = np.column_stack([X,llm_train])
		model_mu = SVC(kernel = 'rbf', C = 1).fit(X_df,Y)
		mu_weighted = model_mu.predict(X_df)
		X_ensemble = np.column_stack([mu_x,mu_weighted])
		preds=model.predict(X_ensemble)
	elif model_type == 'llm-3':
		llm_train = random_sample_predictions(L,pred_cols_list)
		X_df = np.column_stack([X,llm_train])
		model_mu = SVC(kernel = 'rbf', C = 1).fit(X_df,Y)
		mu_weighted = model_mu.predict(X_df)
		X_df_retrain = np.column_stack([X,mu_weighted])
		preds=model.predict(X_df_retrain) 
	elif model_type == 'estimate': 
		preds = model.predict(X)
		
	return preds


def train_sigma2(X=None, Y=None, mu=None, train=False):
	"""
	Computes error terms. 

	Inputs: 
		- X: features
		- Y: ground truth outcomes
		- mu: outcome model (for either treatment condition)
	Outputs: 
		- error term in numerator of objective functino in optimization

	"""

	if train: 
		errs = ((Y-mu)**2).astype(float)
		X_with_intercept = (np.column_stack((np.ones(X.shape[0]), X))).astype(float)
		beta_nnls, _ = nnls(X_with_intercept, errs)
		predicted_errs = np.clip(X_with_intercept @ beta_nnls,0,np.inf)
        
	else: 
		predicted_errs = (Y - mu)**2

	return beta_nnls, predicted_errs


def get_pi_opt(X=None,Z=None,R=None,Y=None, mu=None, sigma2_x=None, e1_x=None,tune_pi=None,budget=None,method=None): 
	"""
	Computes pi values. 
	
	Inputs: 
		- X: covariates 
		- Z: treatment indicator 
		- R: annotation indicator
		- Y: ground truth outcomes
		- mu: outcome model (for either treatment condition)
		- sigma2_x: predicted error term (Y-mu)^2
		- e_x: propensity score model (for either treatment condition)
		- tune_pi: tuning parameter for budget constraint
		- budget: sampling budget (for either treatment condition)
		- method: "opt" using cvxpy solver, "classification" using logistic regression model

	Outputs: 
		- pi: sampling probabilities
	"""
	

	if method == "opt": 
		n = Y.shape[0]
		# Calculate the error term for each data point
		error_values = sigma2_x

		# Define the variable for pi(x, z)
		pi = cp.Variable(n)

		# Define the objective function
		objective = cp.Minimize(cp.sum(cp.multiply(error_values/e1_x,cp.inv_pos(pi))))

		# Define the constraints
		constraints = [
			cp.sum(pi) / n <= budget,  # Budget constraint
			pi >= budget*tune_pi,  # Non-negativity constraint # tune this as a hyperparameter relative to the budget 
			pi <= 1   # Upper bound constraint
		]

		# Set up and solve the problem
		problem = cp.Problem(objective, constraints)
		problem.solve(solver='SCS')

		# Extract the optimal pi values
		pi_opt = pi.value

	elif method == "classification": 
		features = np.column_stack([X,Z])
		clf = LogisticRegression().fit(features, R)
		pi_opt = clf.predict_proba(features)[:,0]
	return pi_opt


def fit_prop_score(X=None, Z=None):
    '''
    Fit a propensity score model to the given data.
    
	Inputs: 
	- X: covariates 
	- Z: treatment indidicator

    Outputs:
		- trained propensity score model
    '''
    
    model_e1 = LogisticRegression(random_state=0, penalty='l2').fit(X, Z)
    
    return model_e1 


# Function to randomly sample from predicted columns and preserve row index
def random_sample_predictions(df, pred_cols, seed=None):
    """
    Randomly samples one value from the specified prediction columns for each row.
    If all values are NaN for a row, NaN is returned for that row.
    
    Inputs:
    - df: DataFrame containing the prediction columns.
    - pred_cols: List of column names to sample from.
    - seed: (Optional) Random seed for reproducibility.

    Outputs:
    - A pandas Series with the randomly sampled predictions.
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Extract the relevant columns as a NumPy array
    pred_array = df[pred_cols].to_numpy()

    # Generate random column indices for each row
    random_indices = np.array([
        np.random.choice(np.where(~np.isnan(row))[0], size=1)[0] if not np.isnan(row).all() else -1
        for row in pred_array
    ])

    # Use the random indices to select values, handling NaNs for rows with no valid predictions
    sampled_predictions = [
        pred_array[i, idx] if idx != -1 else np.nan
        for i, idx in enumerate(random_indices)
    ]
    
    return sampled_predictions


def train_mu(Y=None,X=None,L=None,pred_cols_list=None,method_mu="estimate"):
	'''
	Train outcome model. 
	Inputs: 
		- Y: ground truth outcomes
		- X: covariates 
		- L: llm predictions 
		- pred_cols_list: list of column names for llm predictions to sample from, only use when method_mu is llm-1, llm-2, llm-3 
		- method_mu: method for training outcome model
		  - "estimate" for just using the covariates
		  - "llm-1" for weighted average of covariate and llm predictions
		  - "llm-2" for ensemble of covariates and llm predictions
		  - "llm-3" for retraining outcome model with llm predictions as features
	Outputs:
		- model_mu: trained outcome model 
		- mu: predictions from outcome model 

	'''

	if method_mu == "estimate":
		rf = RandomForestRegressor(max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100,random_state=42)
		model_mu = rf.fit(X,Y)
		mu = model_mu.predict(X)

	elif method_mu == 'llm-1': 
		### 1. Weighted average of LLM predictions only
		rf = RandomForestRegressor(max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100,random_state=42)
		model_mu = rf.fit(X,Y)
		mu_x = model_mu.predict(X)
		llm_train = random_sample_predictions(L,pred_cols_list)
		X_df = np.column_stack([X,llm_train])
		model_llm = SVC(kernel = 'rbf', C = 1).fit(X_df,Y)
		mu_weighted = model_llm.predict(X_df)
		pred_x, pred_xy, y = mu_x,mu_weighted, Y
		pred_x_train, pred_x_test, pred_xy_train, pred_xy_test, y_train, y_test = train_test_split(pred_x, pred_xy, y, test_size=0.2, random_state=42)
		weights = np.linspace(0, 1, 101)
		mses = []
		for w in weights:
			combo_pred = w * pred_x_test + (1 - w) * pred_xy_test
			mses.append(mean_squared_error(y_test, combo_pred))

		best_w = weights[np.argmin(mses)]
		model_mu = 1
		mu = best_w * pred_x + (1 - best_w) * pred_xy

	elif method_mu == 'llm-2': 
		### 2. "isotonic calibration of LLM predictions", train E[Y | LLM predictions ] on the stage 1 data (since the LLM predictions were given covariates in the first place)
		rf = RandomForestRegressor(max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100,random_state=42)
		model_mu = rf.fit(X,Y)
		mu_x = model_mu.predict(X)
		llm_train = random_sample_predictions(L,pred_cols_list)
		X_df = np.column_stack([X,llm_train])
		model_mu = SVC(kernel = 'rbf', C = 1).fit(X_df,Y)
		mu_weighted = model_mu.predict(X_df)
		# 3) Wrap in RandomizedSearchCV for quick tuning
		search = RandomForestRegressor(max_depth=None, min_samples_leaf=4, min_samples_split= 10, n_estimators=200,random_state=42, n_jobs=-1)  # or RandomForestClassifier
		# 4) Fit
		X_ensemble = np.column_stack([mu_x,mu_weighted])
		model_mu = search.fit(X_ensemble,Y)

		# 5) Inspect best params & evaluate
		mu = model_mu.predict(X_ensemble)

	elif method_mu == 'llm-3': 
		# 3. train an outcome model, E[ Y | LLM predictions, covariates}
		llm_train = random_sample_predictions(L,pred_cols_list)
		X_df = np.column_stack([X,llm_train])
		model_mu = SVC(kernel = 'rbf', C = 1).fit(X_df,Y)
		mu_weighted = model_mu.predict(X_df)
		X_df_retrain = np.column_stack([X,mu_weighted])
		rf = RandomForestRegressor(max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=100,random_state=42)
		model_mu = rf.fit(X_df_retrain,Y)
		mu = model_mu.predict(X_df_retrain)
	
	return model_mu, mu 


def _build_riesz_forest():
	"""
	Helper to instantiate ForestRieszATE with standard config.
	Outputs: 
		- ForestRieszATE model with standard configuration for estimating Riesz representer function
	"""
	return ForestRieszATE(criterion='het', n_estimators=1000, min_samples_leaf=2,
                     min_var_fraction_leaf=0.001, min_var_leaf_on_val=True,
                     min_impurity_decrease=0.01, max_samples=.8, max_depth=None,
                     warm_start=False, inference=False, subforest_size=1,
                     honest=True, verbose=0, n_jobs=-1, random_state=123)


def stage_one(Z1,X1,Y1,budget,L1=None,pred_cols_list=None,method_mu='estimate',prop_score='plug-in'): 
	"""
	Stage one of batch adaptive annotation procedure.
	Inputs:	
		- Data: 
			- Z1: treatment vector for stage 1 data
			- X1: covariates for stage 1 data
			- Y1: outcome vector for stage 1 data
			- budget: budget for stage 1 data
			- L1: LLM predictions for stage 1 data
		- Parameters:
			- pred_cols_list: list of column names for llm predictions to sample from, only use when method_mu is llm-1, llm-2, or llm-3
			- method_mu: method for estimating the outcome model
				- "estimate" for just using the covariates
				- "llm-1" for weighted average of covariates and llm predictions
				- "llm-2" for ensemble of covariates and llm predictions
				- "llm-3" for retraining outcome model with llm predictions as features
			- prop_score: method for estimating the propensity score model,
				- "plug-in" for using logistic regression
				- "balance" for using balancing weights
	Outputs:
		- R1_unif: random sample of units to annotate
		- model_mu1: model for estimating the mean function for treatment group 1
		- model_mu0: model for estimating the mean function for treatment group 0
		- model_sigma1: model for estimating the variance function for treatment group 1
		- model_sigma0: model for estimating the variance function for treatment group 0
		- model_e1: model for estimating the propensity score
	"""
	# step 1: random sample R according to kappa
	n1 = Y1.shape[0]
	R1_unif = sampling_r(n=n1, budget=budget)

	# step 2: train \mu1 and \mu0 on data[R==1]
	if method_mu == 'llm-1': 
		model_mu1,mu1 = train_mu(Y=Y1[(Z1 == 1) & (R1_unif == 1)], X=X1[(Z1 == 1) & (R1_unif == 1)],L=L1[(Z1 == 1) & (R1_unif == 1)],pred_cols_list=pred_cols_list, method_mu=method_mu)
		model_mu0,mu0 = train_mu(Y=Y1[(Z1 == 0) & (R1_unif == 1)], X=X1[(Z1 == 0) & (R1_unif == 1)],L=L1[(Z1 == 0) & (R1_unif == 1)],pred_cols_list=pred_cols_list, method_mu=method_mu)
	elif method_mu == 'estimate': 
		model_mu1,mu1 = train_mu(Y=Y1[(Z1 == 1) & (R1_unif == 1)], X=X1[(Z1 == 1) & (R1_unif == 1)], method_mu=method_mu)
		model_mu0,mu0 = train_mu(Y=Y1[(Z1 == 0) & (R1_unif == 1)], X=X1[(Z1 == 0) & (R1_unif == 1)], method_mu=method_mu)
	else: 
		model_mu1,mu1 = train_mu(Y=Y1[(Z1 == 1) & (R1_unif == 1)], X=X1[(Z1 == 1) & (R1_unif == 1)],L=L1[(Z1 == 1) & (R1_unif == 1)],pred_cols_list=pred_cols_list, method_mu=method_mu)
		model_mu0,mu0 = train_mu(Y=Y1[(Z1 == 0) & (R1_unif == 1)], X=X1[(Z1 == 0) & (R1_unif == 1)],L=L1[(Z1 == 0) & (R1_unif == 1)],pred_cols_list=pred_cols_list, method_mu=method_mu)

	# step 3: train \sigma^2(x) on data[R==1]
	model_sigma1,_= train_sigma2(X=X1[(Z1 == 1) & (R1_unif == 1)]*1,Y=Y1[(Z1 == 1) & (R1_unif == 1)],mu=mu1,train=True) 
	model_sigma0,_= train_sigma2(X=X1[(Z1 == 0) & (R1_unif == 1)]*1,Y=Y1[(Z1 == 0) & (R1_unif == 1)],mu=mu0,train=True) 

	# step 4: train propensity score models on data[R==1]
	if prop_score == 'plug-in':
		ZR1 = Z1*R1_unif
		model_e1 = fit_prop_score(X = X1, Z = ZR1)

	elif prop_score == 'balance': 
		X = np.c_[Z1*R1_unif, X1]
		Y1_reshaped = Y1.reshape(-1,1)
		y_scaler = StandardScaler(with_mean=True).fit(Y1_reshaped)
		y = y_scaler.transform(Y1_reshaped)
		est = _build_riesz_forest()
		est.fit(X[:, 1:], X[:, [0]], y.reshape(-1, 1))
		model_e1 = est

	return R1_unif,model_mu1,model_mu0, model_sigma1, model_sigma0, model_e1 


def stage_two(R1=None,Z1=None,Z2=None,X1=None,X2=None,Y1=None,Y2=None,L1=None,L2=None,model_e1=None,model_mu0=None,model_mu1=None,model_sigma0=None,model_sigma1=None,tune_pi=None,budget=None, pred_cols_list=None,method_mu='estimate',prop_score ='plug-in'): 
	"""
	Stage two of batch adaptive annotation procedure.
	Inputs:
		- Data: 
			- R1: annotation indicator for stage 1 data
			- Z1: treatment vector for stage 1 data
			- Z2: treatment vector for stage 2 data
			- X1: covariates for stage 1 data
			- X2: covariates for stage 2 data
			- Y1: outcome vector for stage 1 data
			- Y2: outcome vector for stage 2 data
			- L1: LLM predictions for stage 1 data
			- L2: LLM predictions for stage 2 data
		- Trained models from stage 1 data:
			- model_e1: model for estimating the propensity score
			- model_mu0: model for estimating the mean function for treatment group 0
			- model_mu1: model for estimating the mean function for treatment group 1
			- model_sigma0: model for estimating the variance function for treatment group 0
			- model_sigma1: model for estimating the variance function for treatment group 1
		- Parameters: 
			- tune_pi: tuning parameter for budget constraint in optimization
			- budget: sampling budget for stage 2 data
			- pred_cols_list: list of column names for llm predictions to sample from, only use when method_mu is llm-1, llm-2, or llm-3
			- method_mu: method for estimating the outcome model
				- "estimate" for just using the covariates
				- "llm-1" for weighted average of covariates and llm predictions
				- "llm-2" for ensemble of covariates and llm predictions
				- "llm-3" for retraining outcome model with llm predictions as features
			- prop_score: method for estimating the propensity score model,
				- "plug-in" for using logistic regression
				- "balance" for using balancing weights	
	Outputs:
		- AIPW estimate and variance estimate from the full data (stage 1 and stage 2 combined)
	"""

	# step 5: run optimzation on D1 to obtain pi1,pi0 
	
	if method_mu == 'llm-1': 
		_,mu1=train_mu(Y=Y2[Z2==1], X=X2[Z2 == 1],L=L2[Z2 == 1],pred_cols_list=pred_cols_list, method_mu=method_mu)
		_,mu0=train_mu(Y=Y2[Z2==0], X=X2[Z2 == 0],L=L2[Z2 == 0],pred_cols_list=pred_cols_list, method_mu=method_mu)
	elif method_mu == "estimate":
		_,mu1=train_mu(Y=Y2[Z2==1], X=X2[Z2 == 1],method_mu=method_mu)
		_,mu0=train_mu(Y=Y2[Z2==0], X=X2[Z2 == 0],method_mu=method_mu)
	else: 
		mu1 = predict_func(X=X2[Z2 == 1],L=L2[Z2==1],Y=Y2[Z2==1],pred_cols_list=pred_cols_list,model=model_mu1,model_type=method_mu)
		mu0 = predict_func(X=X2[Z2 == 0],L=L2[Z2==0],Y=Y2[Z2==0],pred_cols_list=pred_cols_list,model=model_mu1,model_type=method_mu)

	
	sigma2_x_mu1 = predict_func(X=X2[Z2==1],model=model_sigma1,model_type='nnls')
	sigma2_x_mu0 = predict_func(X=X2[Z2==0],model=model_sigma0,model_type='nnls')

	if prop_score == 'plug-in':
		e1_x = np.clip(model_e1.predict_proba(X=X2)[:,0],0.01,0.99)
		pi1 = get_pi_opt(Y=Y2[Z2==1], mu=mu1, sigma2_x=sigma2_x_mu1, e1_x=e1_x[Z2==1],tune_pi=tune_pi,budget=budget,method="opt")
		pi0 = get_pi_opt(Y=Y2[Z2==0], mu=mu0, sigma2_x=sigma2_x_mu0, e1_x=(1-e1_x[Z2==0]),tune_pi=tune_pi,budget=budget,method="opt")
	elif prop_score == 'balance': 
		X2 = np.c_[np.ones(X2.shape[0]), X2] # needed to update to match the reisz forest function call
		e1_x,_ = model_e1.predict_riesz_and_reg(X2)
		pi1 = get_pi_opt(Y=Y2[Z2==1], mu=mu1, sigma2_x=sigma2_x_mu1, e1_x=e1_x[Z2==1],tune_pi=tune_pi,budget=budget,method="opt")
		pi0 = get_pi_opt(Y=Y2[Z2==0], mu=mu0, sigma2_x=sigma2_x_mu0, e1_x=(e1_x[Z2==0]),tune_pi=tune_pi,budget=budget,method="opt")
		
	
	pi = np.zeros(Y2.shape[0])
	pi[Z2==1] = np.clip(pi1,0.01,0.99)  
	pi[Z2==0] = np.clip(pi0,0.01,0.99)

	# step 6: Solve for pi2 and sample R2 according to pi2
	kappa = Y1.shape[0]/(Y1.shape[0] + Y2.shape[0])
	pi_2 = 1/(1-kappa)*(pi - kappa*budget)  
	pi_clipped = np.clip(pi_2,0.01,0.99)
	R2 = sampling_r(pi_vals=pi_clipped)

	# step 7: Collect data D1, D2; make D2 = (X2, R2, Z2, R2Y2), full_data = (D1 and D2)
	D1 = pd.DataFrame({"R": R1, "Z": Z1, "Y": Y1, "pi": budget})
	D2 = pd.DataFrame({"R": R2, "Z": Z2, "Y": Y2, "pi": pi_clipped})
	
	if method_mu == 'estimate':
		D1_combined = pd.concat([D1, pd.DataFrame(X1)], axis=1)
		D2_combined = pd.concat([D2, pd.DataFrame(X2)], axis=1)
	else: 
		D1_combined = pd.concat([D1.reset_index(drop=True), pd.DataFrame(X1).reset_index(drop=True), L1.reset_index(drop=True)], axis=1)
		D2_combined = pd.concat([D2.reset_index(drop=True), pd.DataFrame(X2).reset_index(drop=True), L2.reset_index(drop=True)], axis=1)

	full_data = pd.concat([D1_combined, D2_combined], axis=0) 

	X_list = [i for i in range(X1.shape[1])]
	
	sigma2_x_mu1_full = predict_func(X=full_data.loc[full_data['Z'] ==1,X_list],model=model_sigma1,model_type='nnls')
	sigma2_x_mu0_full = predict_func(X=full_data.loc[full_data['Z'] ==0,X_list],model=model_sigma0,model_type='nnls')


	if method_mu == 'llm-1': 
		L_list = L1.columns.to_list()
		_,mu1_full=train_mu(Y=full_data.loc[(full_data['Z'] == 1) & (full_data['R'] == 1),'Y'], X=full_data.loc[(full_data['Z'] == 1) & (full_data['R'] == 1),X_list],L=full_data.loc[(full_data['Z'] == 1) & (full_data['R'] == 1),L_list],pred_cols_list=pred_cols_list, method_mu=method_mu)
		_,mu0_full=train_mu(Y=full_data.loc[(full_data['Z'] == 0) & (full_data['R'] == 1),'Y'], X=full_data.loc[(full_data['Z'] == 0) & (full_data['R'] == 1),X_list],L=full_data.loc[(full_data['Z'] == 0) & (full_data['R'] == 1),L_list],pred_cols_list=pred_cols_list, method_mu=method_mu)		
	elif method_mu == 'estimate':
		model_mu1_full,mu1_full=train_mu(Y=full_data.loc[(full_data['Z'] == 1) & (full_data['R'] == 1),'Y'], X=full_data.loc[(full_data['Z'] == 1) & (full_data['R'] == 1),X_list], method_mu=method_mu)
		model_mu0_full,mu0_full=train_mu(Y=full_data.loc[(full_data['Z'] == 0) & (full_data['R'] == 1),'Y'], X=full_data.loc[(full_data['Z'] == 0) & (full_data['R'] == 1),X_list], method_mu=method_mu)		
	else: 
		L_list = L1.columns.to_list()
		mu1_full = predict_func(X=full_data.loc[full_data['Z'] == 1,X_list],L=full_data.loc[full_data['Z'] == 1,L_list],Y=full_data.loc[full_data['Z'] == 1,'Y'],pred_cols_list=pred_cols_list,model=model_mu1,model_type=method_mu)
		mu0_full = predict_func(X=full_data.loc[full_data['Z'] == 0,X_list],L=full_data.loc[full_data['Z'] == 0,L_list],Y=full_data.loc[full_data['Z'] == 0,'Y'],pred_cols_list=pred_cols_list,model=model_mu0,model_type=method_mu)


	# step 8: Re-estimate propensity score on full data, re-estimate pi on full data (probabilistic classification of R on full data) 
	if prop_score == 'plug-in': 
		ZR = full_data['Z'] * full_data['R']
		model_e1_refit = fit_prop_score(X=full_data[X_list], Z=ZR)
		e1_x_refit = model_e1_refit.predict_proba(X=full_data[X_list])[:,0]
	elif prop_score == 'balance':
		X = np.c_[full_data['Z'] * full_data['R'], full_data[X_list]]
		full_data_y_reshaped = full_data['Y'].to_numpy().reshape(-1,1)
		y_scaler = StandardScaler(with_mean=True).fit(full_data_y_reshaped)
		y = y_scaler.transform(full_data_y_reshaped)
		est = _build_riesz_forest()
		est.fit(X[:, 1:], X[:, [0]], y.reshape(-1, 1))
		e1_x_refit,_ = est.predict_riesz_and_reg(X)


	# step 9: Re-estimate outcome model on RY (observed full-data outcomes) estimate OR LLM 
	# train on R==1
	# repredict on entire dataset
	if method_mu == 'llm-1': 
		_,mu1_refit=train_mu(Y=full_data.loc[full_data['Z'] == 1,'Y'], X=full_data.loc[full_data['Z'] == 1,X_list],L=full_data.loc[full_data['Z'] == 1,L_list],pred_cols_list=pred_cols_list, method_mu=method_mu)
		_,mu0_refit=train_mu(Y=full_data.loc[full_data['Z'] == 0,'Y'], X=full_data.loc[full_data['Z'] == 0,X_list],L=full_data.loc[full_data['Z'] == 0,L_list],pred_cols_list=pred_cols_list, method_mu=method_mu)	
	elif method_mu == 'estimate':
		mu1_refit=predict_func(X=full_data.loc[full_data['Z'] == 1,X_list],model=model_mu1_full, model_type=method_mu)
		mu0_refit=predict_func(X=full_data.loc[full_data['Z'] == 0,X_list], model=model_mu0_full,model_type=method_mu)	
	else: 
		model_mu1_refit,mu1_obs = train_mu(Y=full_data.loc[(full_data['Z'] == 1) & (full_data['R'] == 1),'Y'], X=full_data.loc[(full_data['Z'] == 1) & (full_data['R'] == 1),X_list],L=full_data.loc[(full_data['Z'] == 1)& (full_data['R'] == 1),L_list],pred_cols_list=pred_cols_list, method_mu=method_mu)
		model_mu0_refit,mu0_obs = train_mu(Y=full_data.loc[(full_data['Z'] == 0) & (full_data['R'] == 1),'Y'], X=full_data.loc[(full_data['Z'] == 0) & (full_data['R'] == 1),X_list],L=full_data.loc[(full_data['Z'] == 0)& (full_data['R'] == 1),L_list],pred_cols_list=pred_cols_list, method_mu=method_mu)
		
		mu1_refit = predict_func(X=full_data.loc[full_data['Z'] == 1,X_list],L=full_data.loc[full_data['Z'] == 1,L_list],Y=full_data.loc[full_data['Z'] == 1,'Y'],pred_cols_list=pred_cols_list,model=model_mu1_refit,model_type=method_mu)
		mu0_refit = predict_func(X=full_data.loc[full_data['Z'] == 0,X_list],L=full_data.loc[full_data['Z'] == 0,L_list],Y=full_data.loc[full_data['Z'] == 0,'Y'],pred_cols_list=pred_cols_list,model=model_mu0_refit,model_type=method_mu)


	# step 10: Run the AIPW estimator 
	pi_refit = np.ones(full_data.shape[0])
	point_est, var_pointest = aipw_estimator(Z=full_data['Z'], R=full_data['R'], Y=full_data['Y'], mu1=mu1_refit, mu0=mu0_refit, e1_x=e1_x_refit, pi=pi_refit, prop_score=prop_score)
	return point_est, var_pointest
