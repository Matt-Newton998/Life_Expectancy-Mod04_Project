# Mod04_Life_Expectancy_Project

# Introduction
We are an analysis firm who have been hired by a life insurance firm to find ways to better predict life expectancy.

# Objectives
- Collect data and collate it into one dataset
- Use statistical techniques such as regression analysis (Polnominal, lasso and ridge), elimination of high p-values and removal of outliers
- Optimise model to get a high R^2 value, whilst having appopiate scores in skew, kurtosis and MSE categories. 

# Project Summary
We collected data on factors that may contribute to life expectancy, from our initial dataset of 534 variables we narrowed it down to 14 key variables. 
These key variables were made up of:

 'Injury_deaths_raw_value',
 
 'Teen_births_raw_value',
 
 'Uninsured_adults_raw_value',
 
 'Uninsured_raw_value',
 
 'Air_pollution___particulate_matter_raw_value',
 
 'Diabetes_prevalence_raw_value',
 
 'Sexually_transmitted_infections_raw_value',
 
 'Injury_deaths_raw_value_AND_Teen_births_raw_value',
 
 'Injury_deaths_raw_value_AND_Air_pollution___particulate_matter_raw_value',
 
 'Population_raw_value',
 
 'County_Ranked',
 
 'High_school_graduation_raw_value',
 
 'Teen_births_raw_value_AND_Uninsured_adults_raw_value',
 
 'Alcohol_impaired_driving_deaths_raw_value'

We started off with a baseline model which returned a r^2 of 0.922. After looking at a qq plot we realised that are data was not normally distrubuted. Because of this our we removed the top 10% datapoints from our dataset, this improved the kurtosis and the skewness of our dataset.
We then dropped the datapoints that would be too hard or costly to collect, as this would have real life implicaitons. This returned a r^2 value of 0.766.
We tried eliminating the p-values that were over a alpha of 0.05. This returned a r^2 value of 0.740. As the r^2 value was lower and the kurtosis, skewness, and BIC/AIC values were similar we saw no reason to drop p-values. 

We decided not to use polynominal regression, due to time constraints, the added complexities and the fact that we were satisfied with our linear r^2 value. 

