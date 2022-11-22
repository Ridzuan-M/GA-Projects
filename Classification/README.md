# Project-4-West Nile Virus Modelling Based on Kaggle Competition
Authors: Mark Yung, Ridzuan Mokhtar and Alvin Tan

## Problem Statement

(*From Kaggle*)  
"*Every year from late-May to early-October, public health workers in Chicago setup mosquito traps scattered across the city. Every week from Monday through Wednesday, these traps collect mosquitos, and the mosquitos are tested for the presence of West Nile virus before the end of the week. The test results include the number of mosquitos, the mosquitos species, and whether or not West Nile virus is present in the cohort.*"

We have been tasked by the Chicago Department of Public Health (CDPH) to study the patterns of mosquito propagation, to see whether **usable insights** can be drawn from the mosquito population data in an effort **to effectively predict the movement of the mosquito population in the city over time**.

## Background
The West Nile Virus first appeared in 1937 in Uganda, and subsequently appeared on continental America in 1999, making landfall in New York City. In 2002, North America saw the largest outbreak West Nile meningoencephalitis *anywhere* , with infections stretching from the Mississippi River in the East to the Pacific Coast in the west.

The virus is transmitted primarily by *Culex Pipiens*, otherwise known as the Northern house mosquito. They thrive in areas with many sources of stagnant water where they can easily breed, such as overgrown ponds, poorly managed waste-effluent lagoons, drainage ditches, or just water collected in natural or artificial catchment containers like cans, bird baths, etc.

## Data Cleaning, EDA and Feature Engineering
We will use `train.csv`, `weather.csv` and `spray.csv`for our analysis and modelling. Our key steps include, 
- change Date to datetime format
- drop duplicates in `train.csv` by aggregating NumMosquitos and WnvPresent summed by other columns, since total mosquito recorded in `train.csv`in the column NumMosquitos are capped at 50 rows.
- impute missing data in Tavg, WetBulb, Heat, Cool, Sunrise, Sunset, PrecipTotal, StnPressure and SeaLevel.
- drop columns with insufficient or no data.
- dummify each CodeSum into a separate categorical representer.
- engineer new features such as Daylight(hr), RHumidity and to create cluster s to group numeric columns together (latitude/longitude, wind speed/wind dir and temp/humidity).


## Model
We will apply resampling method Synthetic Minority Over-sampling TEchnique (SMOTENC)  to our imbalanced (~ 95% Wnv not present and ~5% WnvPresent) dataset to obtain a balanced dataset for modelling. The SMOTENC is a resampling technique suitable for use on dataset containing numerical and categorical features. 

We have selected Logistic Regression as the baseline model and using Pycaret we derive Extra Trees, Random Forest, Logistic Regression and Extreme Gradient Boosting as the top model for evaluation and hyperparameter tuning (optimised by 'AUC') in order to derive the best scores. 


| Score              | LR no resampling | LR SMOTE resampling | LR ADASYN resampling | LR SVMSMOTE resampling | LR SMOTENC resampling | LR SMOTENC PCA | xgboost | Extra Trees | Random Forest | CatBoost Classifier |
|--------------------|:----------------:|:-------------------:|:--------------------:|:----------------------:|:---------------------:|:--------------:|:-------:|:-----------:|:-------------:|:-------------------:|
| Acc (train)        |       0.87       |         0.99        |         0.99         |          0.99          |          0.99         |      0.75     |   0.85  |     0.86    |      0.86     |         0.95        |   
| Acc (test)         |       0.75       |         0.73        |         0.73         |          0.73          |          0.74         |      0.73      |   0.74  |     0.78    |      0.79     |         0.90         |   
| MisclassRate(test) |       0.25       |         0.27        |         0.27         |          0.27          |          0.26         |      0.27      |   0.26  |     0.22    |      0.21     |         0.10         |   
| Recall (test)      |       0.69       |         0.68        |         0.68         |          0.69          |          0.68         |      0.68      |   0.75  |     0.73    |      0.68     |         0.45        |   
| Spec (test)        |       0.75       |         0.73        |         0.73         |          0.74          |          0.74         |      0.74      |   0.74  |     0.78    |      0.80      |         0.92        |   
| Precision (test)   |       0.14       |         0.13        |         0.13         |          0.13          |          0.13         |      0.13      |   0.14  |     0.16    |      0.16     |         0.25        |   
| F1 (test)          |       0.23       |         0.22        |         0.22         |          0.22          |          0.22         |      0.22      |   0.24  |     0.26    |      0.26     |         0.32        |   
| ROC_AUC (test)     |       0.72       |         0.71        |         0.71         |          0.71          |          0.71         |      0.71      |   0.74  |     0.76    |      0.74     |         0.68        |

Based on the modelling result, we recommend the Extra Trees model which is our best model for Wnv prediction and the top dominant features identified by the best model are:
    1) Species CULEX RESTAUNS
    2) Daylight(hr)
    3) Month_sin & Month_cos (Cyclical Features Months)
    4) Station
    5) Cluster Lat / Long
    6) Wind Result Speed
    7) Cluster for wind Result Speed & wind Result Direction
    8) Cluster temperature / RHumidity

## Benefit Analysis
We conducted a cost benefit analysis to evaluate if the use of pesticide remains an effective means for mosquito contol and derive an effective plan to deploy pesticides throughout Chicago. Through our analysis, we concluded that since the cost incurred on the pesticide programme is approximately USD301,701 per annum, it would mean that prevention of 15 cases (at its peak in 2005, a total of 225 cases were reported in Chicago) of WNV would make the spray programme in Chicago cost effective. Moreover, Wnv may also have further impact on the workforce and productivity. Hence, a successful pesticide programme will also help minimise the impact to the workforce productivity as a result of man-day losses due to the virus.

## Conclusion
- Highest count of mosquito in Jul and Aug period, which coincides with the period of the highest precipitation (PrecipTotal), relative humidity (RHumidity) and temperature (Tavg).
- Spraying pesticide is an effective mean for mosquito control and prevention of the West Nile Virus. 
- Extra Trees model is selected as our best model and shall be recommended for implementation for Wnv predictions.

## Recommendation
- To enhance the effectiveness of the spray, the spray regime should take into consideration the following
1. Use our model to identify location (positive prediction) to spray pesticide. This will minimise the cost of spraying as compared to spraying for 100% coverage in Chicago.
2. Life cycle of a Culex species mosquito. Since it will take 7 to 10 days for an egg to develop into an adult mosquito, the spray should be performed in the same location every 7 to 10 days.
3. Weather. Based on historical data, mosquito thrives best under hot, humid and rainy condition. Higher spray frequency can be planned and performed during such weather condition.
4. To augment existing spray regime with larvicide regime as a means to limit mosquito breeding and reduce the adult mosquito population density.
