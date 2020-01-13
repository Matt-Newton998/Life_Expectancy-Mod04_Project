
#### !!!! Convert to be used for K-Fold !!!!
# Pre-Processor that 
def preprocess_impute_median(X, y, percentage_test_size = 0.2):
    '''Takes in features and target and implements all preprocessing steps for categorical 
    and continuous features returning train and test DataFrames with targets'''
    
    # Import Libraries
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_squared_log_error
    from sklearn.linear_model import Lasso, Ridge, LinearRegression
    from sklearn.impute import SimpleImputer
    
    # Train-test split (75-25), set seed to 10
    per_test_size = int(percentage_test_size * len(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=per_test_size)
    
    # Remove "object"-type features and SalesPrice from X
    cont_features = [col for col in X.columns if X[col].dtypes in [np.float64, np.int64]]
    X_train_cont = X_train.loc[:, cont_features]
    X_test_cont = X_test.loc[:, cont_features]
    
    # Impute missing values with median using SimpleImputer
    impute = SimpleImputer(strategy='median')
    X_train_imputed = impute.fit_transform(X_train_cont)
    X_test_imputed = impute.fit_transform(X_test_cont)

    # Scale the train and test data
    ss = StandardScaler()
    X_train_imputed_scaled = ss.fit_transform(X_train_imputed)
    X_test_imputed_scaled = ss.transform(X_test_imputed)

    # Create X_cat which contains only the categorical variables
    features_cat = [col for col in X.columns if X[col].dtypes in [np.object]]
    X_train_cat = X_train.loc[:,features_cat]
    X_test_cat = X_test.loc[:,features_cat]

    # Fill nans with a value indicating that that it is missing
    X_train_cat.fillna(value='missing', inplace=True)
    X_test_cat.fillna(value='missing', inplace=True)

    # OneHotEncode Categorical variables
    ohe = OneHotEncoder(handle_unknown='ignore')

    # Transform training and test sets
    X_train_ohe = ohe.fit_transform(X_train_cat)
    X_test_ohe = ohe.transform(X_test_cat)

    # Convert these columns into a DataFrame
    columns = ohe.get_feature_names(input_features=X_train_cat.columns)
    cat_train_df = pd.DataFrame(X_train_ohe.todense(), columns=columns)
    cat_test_df = pd.DataFrame(X_test_ohe.todense(), columns=columns)
    
    # Combine categorical and continuous features into the final dataframe
    X_train_all = pd.concat([pd.DataFrame(X_train_imputed_scaled), cat_train_df], axis=1)
    X_test_all = pd.concat([pd.DataFrame(X_test_imputed_scaled), cat_test_df], axis=1)
    
    return X_train_all, X_test_all, y_train, y_test


# Linear Regression Feature Combination Selection
def feature_combinations_r_sqrd(X, y, k_splt=10, num_feat_comb=2):
    # Imports
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from itertools import combinations
    
    # Create Regression & Combinations
    regression = LinearRegression()
    combinations = list(combinations(list(X.columns), num_feat_comb))
    
    # Create cross-validation & output a bassline MSE score 
    crossvalidation = KFold(n_splits=k_splt, shuffle=True, random_state=1)
    baseline = np.mean(cross_val_score(regression, X, y, scoring='r2', cv=crossvalidation))
    print("Baseline:", round(baseline, 3))
    
    # Create cross-validation & output a bassline MSE score as a DataFrame
    var_1 = []
    var_2 = []
    comb_scores = []
    data = X.copy()
    
    for comb in combinations:
        data['interaction'] = data[comb[0]] * data[comb[1]]
        score = np.mean(cross_val_score(regression, data, y, scoring='r2', cv=crossvalidation))
        if score > baseline: 
            var_1.append(comb[0])
            var_2.append(comb[1])
            comb_scores.append(round(score,3))
    
    df_base = pd.DataFrame(data=[var_1, var_2, comb_scores])
    df_base = df_base.T  
    df_base.rename(columns={0: "var_1", 1: "var_2", 2: "scores"}, inplace=True)
    df_base.sort_values(by='scores', inplace = True, ascending=False)
    
    return df_base



def feature_combinations_r_sqrd_with_Inter_df(X, y, k_splt=10, num_feat_comb=2):
    # Imports
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from itertools import combinations
    
    # Create Regression & Combinations
    regression = LinearRegression()
    combinations = list(combinations(list(X.columns), num_feat_comb))
    
    # Create cross-validation & output a bassline MSE score 
    crossvalidation = KFold(n_splits=k_splt, shuffle=True, random_state=1)
    baseline = np.mean(cross_val_score(regression, X, y, scoring='r2', cv=crossvalidation))
    print("Baseline:", round(baseline, 3))
    
    # Create cross-validation & output a bassline MSE score as a DataFrame
    comb_scores = []
    inter_cols = []
    inter_score = []
    data = X.copy()
    
    for comb in combinations:
        data['interaction'] = data[comb[0]] * data[comb[1]]
        score = np.mean(cross_val_score(regression, data, y, scoring='r2', cv=crossvalidation))
        if score > baseline: 
            comb_scores.append(round(score,3))
            inter_cols.append((str(comb[0]) + '_' + str(comb[1])))
            inter_score.append(data[comb[0]] * data[comb[1]])
    
    df_base = pd.DataFrame(data=[inter_cols, comb_scores])
    df_base = df_base.T  
    df_base.rename(columns={0: "Interaction", 1: "CV_score"}, inplace=True)
    df_base.sort_values(by='CV_score', inplace = True, ascending=False )
    df_base.reset_index(drop=True, inplace = True)
    
    df_interactions_scores = pd.DataFrame(data=inter_score , index=inter_cols)
    df_interactions_scores = df_interactions_scores.T
    
    return df_base , df_interactions_scores


def add_interaction_feature(data, df_inter, df_score, num_inter):
    i=0
    
    while i < num_inter:
        col = df_inter['Interaction'][i]
        data[col] = df_score[col]
        i+=1
    
    return data




import statsmodels.formula.api as smf


# Linear forward selection functions
def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model