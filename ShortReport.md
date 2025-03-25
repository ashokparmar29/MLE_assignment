**Report on Data Preprocessing, Dimensionality Reduction, and Model Evaluation**

## 1. Data Preprocessing
### Steps and Rationale:
- **Data Loading and Inspection:** The dataset was examined for anomalies, revealing no major issues in the initial features.
- **Feature Correlation Analysis:** The first 20 features exhibited high correlation (coefficient > 0.8), indicating potential redundancy.
- **Target Variable Analysis:** A boxplot suggested possible outliers in DON concentration values.
- **Outlier Removal:** The normal DON concentration range for healthy foods is 0–5000, with 98th percentile (20060) set as the threshold to remove extreme values.
- **Standardization:** Standard scaling was applied to normalize feature values. (Found that it is reducing performance so removed from final pipeline)
- **Feature Engineering:**
  - 183 spectral bands had weak correlation (|r| < 0.1) with DON concentration, suggesting limited predictive value.
  - No features were found to have extremely low variance (< 0.001), ensuring all retained features had some variability.

## 2. Dimensionality Reduction
### PCA Insights:
- Principal Component Analysis (PCA) was used to analyze variance.
- The explained variance plot showed that a subset of components retained most of the information while reducing dimensionality.
- Reducing the number of features helped mitigate redundancy and computational complexity.

## 3. Model Selection, Training, and Evaluation
- **Model Selection:** Multiple models were evaluated for performance.
- **Training:** Models were trained on the preprocessed dataset, leveraging feature reduction techniques.
- **Evaluation Metrics:**
  - Performance was assessed using standard metrics such as RMSE and R-squared.
  - The final model demonstrated competitive predictive accuracy.

## 4. Key Findings and Possible Improvements
### Key Findings:
- PCA effectively reduced dimensionality while retaining key information.
- Feature selection, additional features computation and outlier handling improved model stability.
- The final model achieved strong predictive performance based on evaluation metrics.
- Standardization of features is reducing the performance. Without standardization, the model gives good performance. 
### Suggested Improvements:
- Investigate alternative feature selection methods beyond PCA.
- Further Fine-tune hyperparameters using automated search techniques (e.g., Grid Search, Bayesian Optimization).
- Explore deep learning approaches.
- Reduce complexity.



