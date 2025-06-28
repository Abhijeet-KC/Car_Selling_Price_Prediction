<body>

<h1> Car Selling Price Prediction</h1>

<p>This project focuses on building machine learning models to accurately predict the selling price of used cars. It explores data preprocessing, feature engineering, regularization techniques, model comparison, and hyperparameter tuning using cross-validation.</p>

<h2> Project Workflow</h2>

<h3> 1. Exploratory Data Analysis (EDA)</h3>
<ul>
  <li>Removed missing and duplicate rows</li>
  <li>Plotted distributions of numeric and categorical features</li>
  <li>Checked correlations and multicollinearity</li>
</ul>

<h3> 2. Data Preprocessing</h3>
<ul>
  <li>Log-transformed skewed features: <code>selling_price</code>, <code>km_driven</code>, <code>engine</code></li>
  <li>Handled outliers using IQR and percentile methods</li>
  <li>One-hot encoded categorical features</li>
  <li>Standardized numerical values using <code>StandardScaler</code></li>
  <li>Built preprocessing pipelines using <code>ColumnTransformer</code></li>
</ul>

<h3> 3. Feature Engineering</h3>
<ul>
  <li>Derived new features:
    <ul>
      <li><code>car_age = current_year - year</code></li>
      <li><code>price_per_km = selling_price / km_driven</code></li>
      <li><code>engine_per_seat = engine / seats</code></li>
    </ul>
  </li>
  <li>In some experiments, source columns like <code>year</code>, <code>km_driven</code>, <code>engine</code> were dropped to observe the impact</li>
</ul>

<h3> 4. Model Training & Evaluation</h3>
<p>All models were trained using pipelines and evaluated using cross-validation.</p>
<ul>
  <li>Linear Regression (with and without L1/L2 regularization)</li>
  <li>Random Forest Regressor</li>
  <li>XGBoost Regressor</li>
  <li>LightGBM Regressor</li>
</ul>

<h4> Regularization:</h4>
<ul>
  <li><b>Lasso (L1)</b> used for automatic feature selection</li>
  <li><b>Ridge (L2)</b> used to reduce coefficient variance</li>
  <li><b>ElasticNet (L1 + L2)</b> found best parameters:
    <pre>
    {'model__alpha': 0.01, 'model__l1_ratio': 0.2}
    Best R²: 0.8132
    </pre>
  </li>
</ul>

<h3> 5. Outlier Impact</h3>

<h4> With Outliers:</h4>
<ul>
  <li>Linear Regression: MAE = 101,229.81, R² = 0.714</li>
  <li>Random Forest: MAE = 90,691.03, R² = 0.825</li>
  <li>XGBoost: MAE = 89,614.51, R² = 0.846</li>
  <li>LightGBM: MAE = 90,948.43, R² = 0.846</li>
</ul>

<h4> Without Outliers:</h4>
<ul>
  <li>Linear Regression: MAE = 80,589.06, R² = 0.849</li>
  <li>Random Forest: MAE = 71,006.79, R² = 0.860</li>
  <li>XGBoost: MAE = 72,881.67, R² = 0.866</li>
  <li>LightGBM: MAE = 73,807.30, R² = 0.845</li>
</ul>

<h3> Tuned XGBoost Model</h3>
<ul>
  <li>Best parameters via GridSearchCV:
    <pre>{'model__learning_rate': 0.2, 'model__max_depth': 5, 'model__n_estimators': 200}</pre>
  </li>
  <li>Best CV R²: 0.9930</li>
  <li><b>Test Set:</b> R² = 0.9933, MAE = 11480.94</li>
  <li>Saved as <code>xgboost_pipeline_tuned.pkl</code></li>
</ul>

<h3> Final Evaluation (Best Model)</h3>
<table border="1" cellpadding="6">
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>MAE</td><td>11480.94</td></tr>
  <tr><td>MSE</td><td>284920425.62</td></tr>
  <tr><td>RMSE</td><td>16879.59</td></tr>
  <tr><td>R²</td><td>0.9933</td></tr>
  <tr><td>MAPE</td><td>3.15%</td></tr>
</table>

<h2> Experiments with Feature Engineering</h2>

<h4> When only new features were used and original columns dropped:</h4>
<ul>
  <li>Test R²: 0.8577</li>
  <li>MAPE: 15.24%</li>
</ul>

<h4> When both original + derived features were used:</h4>
<ul>
  <li>Test R²: ~0.9933</li>
  <li>MAPE: 3.15%</li>
</ul>

<p><strong>Conclusion:</strong> Keeping both raw and derived features provided the highest predictive power. Removing source columns (like <code>year</code>, <code>engine</code>) decreased accuracy.</p>

<h2> How to Run</h2>
<pre>
pip install -r requirements.txt
</pre>

<pre><code>import joblib
model = joblib.load("models/xgboost_pipeline_tuned.pkl")
predictions = model.predict(X_test)
</code></pre>

<h2> Future Enhancements</h2>
<ul>
  <li>Use <code>Optuna</code> or <code>BayesianOptimization</code> for smarter tuning</li>
  <li>Deploy using Flask or Streamlit</li>
  <li>Visualize SHAP or LIME for explainable AI</li>
  <li>Try stacking or ensembling models</li>
</ul>

<h2> Credits</h2>
<p>Developed by <strong>Abhijeet K.C.</strong> during the ML Project (2025). Dataset: <code>car.csv</code></p>

<p> Happy Coding! </p>
</body>

