# Customer Churn Analysis: Jupyter Notebook Portfolio

## Overview

This directory contains a comprehensive suite of Jupyter notebooks implementing advanced data science methodologies for customer churn analysis and prediction. Each notebook follows modern data science best practices with professional documentation, rigorous statistical validation, and production-ready implementations.

## Notebook Architecture

### 00. Master Analysis Overview (`00_master_analysis_overview.ipynb`)
**Central navigation hub and project controller**
- Complete project architecture documentation
- Notebook execution sequence and dependencies
- Technical requirements and environment setup
- Runtime estimates and resource requirements

### 01. Exploratory Data Analysis (`01_exploratory_data_analysis.ipynb`)
**Comprehensive statistical profiling and feature analysis**
- Automated data quality assessment with custom metrics classes
- Advanced correlation analysis with statistical significance testing
- Distribution analysis with normality tests and outlier detection
- Feature engineering recommendations based on statistical properties
- Interactive visualizations using plotly and seaborn

**Key Technical Features:**
- Custom `DataQualityMetrics` dataclass for structured assessment
- Automated missing value analysis with imputation strategies
- Cramér's V correlation for categorical variables
- Mutual information analysis for feature relevance

### 02. Customer Segmentation (`02_customer_segmentation.ipynb`)
**Advanced unsupervised learning for customer profiling**
- Multi-algorithm clustering: K-Means, DBSCAN, Hierarchical, Gaussian Mixture
- Optimal cluster determination using Elbow Method, Silhouette Analysis, Gap Statistic
- Dimensionality reduction with PCA, t-SNE, and UMAP visualization
- Statistical validation using Calinski-Harabasz and Davies-Bouldin indices
- Business interpretation with segment profiling and marketing strategy development

**Key Technical Features:**
- Yellowbrick integration for clustering visualization
- Custom cluster validation metrics
- Advanced preprocessing pipelines with StandardScaler and RobustScaler
- Multi-dimensional visualization framework

### 03. Churn Prediction Modeling (`03_churn_prediction_modeling.ipynb`)
**State-of-the-art machine learning implementation**
- Algorithm portfolio: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, Neural Networks
- PyTorch implementation with custom neural network architectures
- Hyperparameter optimization using Optuna Bayesian optimization
- Cross-validation with stratified k-fold and temporal consistency
- Model interpretability using SHAP analysis and LIME explanations

**Key Technical Features:**
- Custom PyTorch Dataset and DataLoader implementations
- Ensemble methods with VotingClassifier and StackingClassifier
- Advanced evaluation metrics including AUC-ROC, AUC-PR, and business metrics
- Feature importance analysis with multiple methodologies

### 04. Survival Analysis (`04_survival_analysis.ipynb`)
**Time-to-event modeling for customer retention**
- Kaplan-Meier estimation for non-parametric survival analysis
- Cox Proportional Hazards modeling with covariate effects
- Parametric survival models: Weibull, Log-Normal, Exponential
- Proportional hazards testing using Schoenfeld residuals
- Customer lifetime value integration with survival probabilities

**Key Technical Features:**
- Lifelines library implementation with custom utilities
- Right-censoring handling for active customers
- Concordance index and integrated Brier score validation
- Hazard ratio interpretation with confidence intervals

### 05. A/B Testing Framework (`05_ab_testing_framework.ipynb`)
**Statistical experimentation for retention strategies**
- Power analysis and sample size determination for retention experiments
- Monte Carlo simulation of intervention strategies
- Bayesian A/B testing with PyMC implementation
- Multiple testing correction using Bonferroni and FDR methods
- ROI analysis with confidence interval estimation

**Key Technical Features:**
- Custom power analysis functions with effect size sensitivity
- Bayesian posterior probability estimation
- Advanced statistical testing with appropriate corrections
- Business impact quantification with financial modeling

### 06. Cost-Sensitive Modeling (`06_cost_sensitive_modeling.ipynb`)
**Business-optimized machine learning approaches**
- Cost matrix development with customer acquisition and retention costs
- Threshold optimization for profit maximization
- Cost-sensitive algorithm variants using costcla library
- Expected value calculation with uncertainty quantification
- Sensitivity analysis across different business scenarios

**Key Technical Features:**
- Custom cost-sensitive evaluation metrics
- Mathematical optimization of decision thresholds
- Integration with customer lifetime value calculations
- Robust financial modeling with Monte Carlo simulation

### 07. Model Comparison and Evaluation (`07_model_comparison_evaluation.ipynb`)
**Comprehensive model benchmarking and selection**
- Standardized evaluation across multiple performance metrics
- Statistical testing for model comparison using McNemar's test
- Model calibration analysis with reliability diagrams
- Fairness assessment with demographic parity evaluation
- Production readiness assessment with stability analysis

**Key Technical Features:**
- Comprehensive metric portfolio including business-specific KPIs
- Model calibration using CalibratedClassifierCV
- Statistical significance testing for performance differences
- Advanced visualization with radar charts and comparison matrices

### 08. Business Insights and Recommendations (`08_business_insights_recommendations.ipynb`)
**Strategic synthesis and actionable business intelligence**
- Integration of findings from all analytical components
- ROI quantification for proposed retention strategies
- Implementation roadmap with prioritized action items
- KPI framework for ongoing performance monitoring
- Risk assessment and mitigation strategy development

**Key Technical Features:**
- Financial modeling with numpy-financial integration
- Strategic priority matrix development
- Customer value optimization algorithms
- Performance monitoring dashboard specifications

## Technical Requirements

### Environment Setup
```bash
# Create isolated environment
python -m venv churn_analysis_env
source churn_analysis_env/bin/activate  # Linux/Mac
churn_analysis_env\Scripts\activate     # Windows

# Install all dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### System Requirements
- **Python**: 3.8+
- **RAM**: Minimum 8GB, Recommended 16GB
- **CPU**: Multi-core processor for parallel processing
- **Storage**: 5GB free space for intermediate files
- **GPU**: Optional for PyTorch deep learning models

### Execution Sequence
1. Start with `00_master_analysis_overview.ipynb` for project orientation
2. Execute notebooks in numerical order (01-08)
3. Allow 2-3 hours total runtime for complete analysis
4. Each notebook can be run independently after prerequisites

## Advanced Features

### Professional Implementation Standards
- Type hints and dataclass structures for code clarity
- Comprehensive error handling and logging
- Modular design with reusable utility functions
- Production-ready code with performance optimization

### Statistical Rigor
- Appropriate hypothesis testing with multiple comparison corrections
- Bootstrap confidence intervals for robust uncertainty quantification
- Cross-validation with proper temporal consistency
- Advanced statistical validation metrics

### Business Integration
- Direct connection between statistical findings and business metrics
- Financial modeling with realistic cost structures
- Actionable recommendations with quantified ROI projections
- Strategic prioritization based on statistical evidence

### Visualization Excellence
- Interactive plotly visualizations for stakeholder engagement
- Professional matplotlib/seaborn styling
- Multi-dimensional data representation
- Publication-ready figure generation

## Output Artifacts

Each notebook generates:
- **Statistical Reports**: Comprehensive analysis summaries
- **Visualizations**: High-quality plots and interactive charts
- **Model Artifacts**: Serialized models for production deployment
- **Business Recommendations**: Actionable strategy documents
- **Performance Metrics**: Detailed evaluation results

## Integration with Existing Project

These notebooks complement the existing project structure:
- **Scripts Integration**: Notebooks reference existing utility functions
- **Data Pipeline**: Uses established preprocessing workflows  
- **Model Compatibility**: Compatible with existing trained models
- **Documentation**: Extends existing analysis documentation

## Best Practices Implemented

1. **Reproducibility**: Fixed random seeds and version-controlled dependencies
2. **Modularity**: Reusable functions and clear separation of concerns
3. **Documentation**: Comprehensive markdown explanations and code comments
4. **Validation**: Multiple validation approaches for robust results
5. **Business Focus**: Direct connection between analysis and business value

