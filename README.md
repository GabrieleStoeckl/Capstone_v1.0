# Increasing Brand Penetration by Acquiring New Customers: A Prediction of New Product Purchases in FMCG

This data science project aims at predicting new product purchases using different ML classification models and understanding drivers by calculating importances via SHAP, based on a comprehensive dataset for the yoghurt product category with transaction, household, and shopper marketing data.

## Contents

[Data understanding notebooks](https://github.com/GabrieleStoeckl/Capstone_v1.0/tree/main/01_data_understanding): 
* [Comparing different product categories in order to select elegible product category](https://github.com/GabrieleStoeckl/Capstone_v1.0/blob/main/01_data_understanding/Comparing_product_categories.ipynb)

[Data preparation notebooks](https://github.com/GabrieleStoeckl/Capstone_v1.0/tree/main/02_data_preparation): 
* [Module for preparing category datasets](https://github.com/GabrieleStoeckl/Capstone_v1.0/blob/main/02_data_preparation/category_data_builder.py)
* [Data used for modelling](https://github.com/GabrieleStoeckl/Capstone_v1.0/blob/main/02_data_preparation/yoghurt_w_dummy_data.csv)

[EDA](https://github.com/GabrieleStoeckl/Capstone_v1.0/tree/main/03_EDA)

[Modelling](https://github.com/GabrieleStoeckl/Capstone_v1.0/tree/main/04_modelling):
* [Benchmark/dummy model and DecisionTreeClassifier](https://github.com/GabrieleStoeckl/Capstone_v1.0/blob/main/04_modelling/yoghurt_benchmark_and_clf_final.ipynb)
* [Random Forest](https://github.com/GabrieleStoeckl/Capstone_v1.0/blob/main/04_modelling/yoghurt_rf_final.ipynb)
* [XGBoost](https://github.com/GabrieleStoeckl/Capstone_v1.0/blob/main/04_modelling/yoghurt_xgboost_final.ipynb)
* [LightGBM](https://github.com/GabrieleStoeckl/Capstone_v1.0/blob/main/04_modelling/yoghurt_lightgbm_final.ipynb)
* [StackingClassifier](https://github.com/GabrieleStoeckl/Capstone_v1.0/blob/main/04_modelling/yoghurt_stacking_final.ipynb)
* [Inderstanding importances with SHAP](https://github.com/GabrieleStoeckl/Capstone_v1.0/blob/main/04_modelling/yoghurt_importances_with_shap.ipynb)

[Project report](https://github.com/GabrieleStoeckl/Capstone_v1.0/blob/main/Project_report_final.pdf)

## Data
The data is an anonymized dataset provided by dunnhumby and can be obtained from https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey).
[dunnhumby - The Complete Journey User Guide](https://github.com/GabrieleStoeckl/Capstone_v1.0/blob/main/dunnhumby%20-%20The%20Complete%20Journey%20User%20Guide.pdf) provides a good overview and explanation.

## Abstract
According to the Ehrenberg Bass marketing philosophy, attracting new customers is crucial for brands' sustained business success. In the highly competitive FMCG market often shopper marketing is leveraged to this end. The objective of this project was to develop a model that predicts product switching within the yoghurt category based on transaction data from 800 households collected over a period of two years for a specific
retailer as well as the corresponding shopper marketing data. Primarily, Random Forest, XGBoost and LightGBM classification models were used. Best-performing models were fed into a StackingClassifier which provided the basis for explaining feature importance via SHAP values. The resulting model's predictive power is satisfactory, especially given the limited nature of available features for this project. 

| Model       | Benchmark   | Random Forest | XGBoost     | LightGBM    | StackingClassifier |
| ----------- | ----------- | ------------- | ----------- | ----------- | ------------------ |
| f1          | 0.3342      | 0.6660        | 0.6609      | 0.6441      | 0.6741             |

Pricing and household demographics were identified as key levers for triggering new product purchases. Further research building on this work ought to aim at including features covering a broader set of marketing strategies to further increase model performance.

## Contributing and contact
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please feel free to contact me at gabrielestoeckl@web.de if you have questions or would like to discuss anything.



