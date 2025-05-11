import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(X, categorical_columns=None, numerical_columns=None):
    """
    Preprocess data for machine learning.
    
    Parameters:
    X (DataFrame): Input features DataFrame
    categorical_columns (list): List of categorical column names
    numerical_columns (list): List of numerical column names
    
    Returns:
    ColumnTransformer: Preprocessor for the data
    """
    try:
        if X is None or len(X) == 0:
            return None
        
        # Identify column types if not specified
        if categorical_columns is None and numerical_columns is None:
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
        
        # Create transformers
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_columns),
                ('num', numerical_transformer, numerical_columns)
            ]
        )
        
        return preprocessor
    
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        return None

def create_predictive_model(X, y, model_type='random_forest', problem_type='classification', 
                           test_size=0.2, random_state=42, **model_params):
    """
    Create and train a predictive model.
    
    Parameters:
    X (DataFrame): Feature DataFrame
    y (Series): Target variable
    model_type (str): Type of model ('random_forest', 'logistic_regression', 'svm', 'decision_tree')
    problem_type (str): 'classification' or 'regression'
    test_size (float): Proportion of data to use for testing
    random_state (int): Random seed
    model_params (dict): Additional parameters for the model
    
    Returns:
    tuple: (model, X_test, y_test, y_pred, metrics)
    """
    try:
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            logger.warning("Empty or None input data for model creation")
            return None, None, None, None, None
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Identify column types
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
        
        # Create preprocessor
        preprocessor = preprocess_data(X, categorical_columns, numerical_columns)
        
        # Select the model based on model_type and problem_type
        if problem_type == 'classification':
            if model_type == 'random_forest':
                model = RandomForestClassifier(random_state=random_state, **model_params)
            elif model_type == 'logistic_regression':
                model = LogisticRegression(random_state=random_state, **model_params)
            elif model_type == 'svm':
                model = SVC(random_state=random_state, probability=True, **model_params)
            elif model_type == 'decision_tree':
                model = DecisionTreeClassifier(random_state=random_state, **model_params)
            else:
                logger.warning(f"Unknown classification model type: {model_type}. Using Random Forest.")
                model = RandomForestClassifier(random_state=random_state)
        
        elif problem_type == 'regression':
            if model_type == 'random_forest':
                model = RandomForestRegressor(random_state=random_state, **model_params)
            elif model_type == 'linear_regression':
                model = LinearRegression(**model_params)
            elif model_type == 'svm':
                model = SVR(**model_params)
            elif model_type == 'decision_tree':
                model = DecisionTreeRegressor(random_state=random_state, **model_params)
            else:
                logger.warning(f"Unknown regression model type: {model_type}. Using Random Forest.")
                model = RandomForestRegressor(random_state=random_state)
        
        else:
            logger.warning(f"Unknown problem type: {problem_type}. Using classification.")
            model = RandomForestClassifier(random_state=random_state)
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        if problem_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        else:
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_test, y_pred)
        
        return pipeline, X_test, y_test, y_pred, metrics
    
    except Exception as e:
        logger.error(f"Error creating predictive model: {str(e)}")
        return None, None, None, None, None

def feature_importance(model, feature_names):
    """
    Extract feature importance from a model.
    
    Parameters:
    model: Trained model with feature_importances_ attribute
    feature_names (list): List of feature names
    
    Returns:
    DataFrame: Feature importance DataFrame
    """
    try:
        if model is None or not hasattr(model, 'named_steps'):
            logger.warning("Invalid model for feature importance extraction")
            return None
        
        # Get model from pipeline
        if 'model' in model.named_steps:
            model_step = model.named_steps['model']
        else:
            model_step = model
        
        # Check if model has feature_importances_ attribute
        if not hasattr(model_step, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return None
        
        # Get feature importance
        importances = model_step.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    except Exception as e:
        logger.error(f"Error extracting feature importance: {str(e)}")
        return None

def generate_prediction_map(model, feature_gdf, x_min, y_min, x_max, y_max, resolution=100):
    """
    Generate a prediction map using a trained model.
    
    Parameters:
    model: Trained model pipeline
    feature_gdf (GeoDataFrame): GeoDataFrame with feature data
    x_min, y_min, x_max, y_max (float): Bounds for the prediction area
    resolution (int): Number of grid points in each dimension
    
    Returns:
    tuple: (x_grid, y_grid, prediction_values)
    """
    try:
        if model is None or feature_gdf is None:
            logger.warning("Invalid model or feature data for prediction map generation")
            return None, None, None
        
        # Create grid
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Create points for prediction
        points = []
        for i in range(resolution):
            for j in range(resolution):
                points.append(Point(X[i, j], Y[i, j]))
        
        # Create GeoDataFrame for points
        points_gdf = gpd.GeoDataFrame({'geometry': points}, crs=feature_gdf.crs)
        
        # Extract features for each point
        # This step would typically involve spatial joins or attribute extraction
        # For simplicity, we'll create dummy features - in a real application,
        # this would be replaced with actual feature extraction
        
        # Create a new DataFrame for features at each grid point
        features_df = pd.DataFrame(index=range(len(points)))
        
        # Extract features from spatial data
        # Example: Distance to nearest fault line
        if 'fault' in feature_gdf.columns or 'FAULT' in feature_gdf.columns:
            fault_col = 'fault' if 'fault' in feature_gdf.columns else 'FAULT'
            features_df['distance_to_fault'] = np.random.uniform(0, 1, len(points))  # Placeholder
        
        # Example: Lithology type (categorical)
        if 'lithology' in feature_gdf.columns or 'LITHOLOGY' in feature_gdf.columns:
            lith_col = 'lithology' if 'lithology' in feature_gdf.columns else 'LITHOLOGY'
            lith_types = feature_gdf[lith_col].unique()
            features_df['lithology'] = np.random.choice(lith_types, len(points))  # Placeholder
        
        # Add more features as needed
        # ...
        
        # For demonstration, add some synthetic features
        features_df['feature1'] = np.sin(X.flatten() * 0.1) * np.cos(Y.flatten() * 0.1)
        features_df['feature2'] = np.abs(X.flatten() - (x_max + x_min) / 2) + np.abs(Y.flatten() - (y_max + y_min) / 2)
        features_df['feature3'] = np.random.normal(0, 1, len(points))
        
        # Make predictions
        try:
            predictions = model.predict_proba(features_df)[:, 1]  # For binary classification, get probability of class 1
        except:
            try:
                predictions = model.predict(features_df)  # For regression or if predict_proba fails
            except Exception as e:
                logger.error(f"Error making predictions: {str(e)}")
                return X, Y, np.zeros((resolution, resolution))
        
        # Reshape predictions to grid
        prediction_grid = predictions.reshape(resolution, resolution)
        
        return X, Y, prediction_grid
    
    except Exception as e:
        logger.error(f"Error generating prediction map: {str(e)}")
        return None, None, None

def evaluate_model_cv(model, X, y, cv=5, scoring='accuracy'):
    """
    Evaluate model using cross-validation.
    
    Parameters:
    model: Model or pipeline
    X (DataFrame): Feature DataFrame
    y (Series): Target variable
    cv (int): Number of cross-validation folds
    scoring (str): Scoring metric
    
    Returns:
    dict: Cross-validation scores
    """
    try:
        if model is None or X is None or y is None:
            logger.warning("Invalid inputs for cross-validation")
            return None
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        # Calculate statistics
        result = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'min_score': cv_scores.min(),
            'max_score': cv_scores.max(),
            'scores': cv_scores.tolist()
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        return None

def plot_confusion_matrix(conf_matrix, class_names=None, figsize=(8, 6)):
    """
    Plot a confusion matrix.
    
    Parameters:
    conf_matrix (array): Confusion matrix
    class_names (list): Class names
    figsize (tuple): Figure size
    
    Returns:
    matplotlib.figure.Figure: Confusion matrix figure
    """
    try:
        if conf_matrix is None:
            logger.warning("No confusion matrix provided for plotting")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No confusion matrix data available",
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set class names
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(conf_matrix))]
        
        # Plot confusion matrix
        import seaborn as sns
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names)
        
        # Set labels
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")
        # Return a basic figure with error message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Error creating confusion matrix plot: {str(e)}",
               horizontalalignment='center', verticalalignment='center')
        return fig

def create_synthetic_training_data(num_samples=1000, num_features=5, classification=True):
    """
    Create synthetic training data for demonstration purposes.
    
    Parameters:
    num_samples (int): Number of samples to generate
    num_features (int): Number of features to generate
    classification (bool): Whether to create classification or regression targets
    
    Returns:
    tuple: (X, y) - feature DataFrame and target
    """
    try:
        # Create synthetic feature data
        X = pd.DataFrame()
        
        # Create numeric features
        for i in range(num_features - 2):
            X[f'feature_{i+1}'] = np.random.normal(0, 1, num_samples)
        
        # Create categorical features
        categories = ['Type A', 'Type B', 'Type C', 'Type D']
        X['category_1'] = np.random.choice(categories, num_samples)
        X['category_2'] = np.random.choice(categories, num_samples)
        
        # Create target variable
        if classification:
            # Create a classification target based on features
            logits = 0.5 * X['feature_1'] - 0.2 * X['feature_2'] + np.random.normal(0, 0.5, num_samples)
            probabilities = 1 / (1 + np.exp(-logits))
            y = (probabilities > 0.5).astype(int)
        else:
            # Create a regression target based on features
            y = 2.0 * X['feature_1'] - 1.5 * X['feature_2'] + 0.5 * X['feature_3'] + np.random.normal(0, 0.5, num_samples)
        
        return X, y
    
    except Exception as e:
        logger.error(f"Error creating synthetic data: {str(e)}")
        return None, None
