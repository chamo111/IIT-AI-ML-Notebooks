# IIT-AI-ML-Notebooks
A curated set of notebooks developed during my IIT course, covering fundamental concepts, coding practice, and applied machine learning experiments.

File: 1. Titanic_DT.ipynb

This notebook explores the Titanic dataset and builds a Decision Tree Classifier to predict passenger survival. It includes a complete end-to-end machine learning workflow, covering everything from data preprocessing to model evaluation and visualization.

Key Steps Covered
- Dataset loading and exploration using Seaborn and Pandas
- Data cleaning and preprocessing
- Handling missing values (Age, Embarked, Deck)
- Descriptive statistics and distribution checks
- Feature engineering and dataset preparation
- Train–test split for model evaluation
- Decision Tree model training using DecisionTreeClassifier
- Model evaluation (accuracy score, classification report, confusion matrix)
- Decision tree visualization using plot_tree

Outcome
This notebook demonstrates how to clean a real dataset and train a simple, interpretable machine learning model. It acts as a solid starting point for building more advanced models in future notebooks.


File: 2. HousePricePrediction.ipynb

This notebook builds a Linear Regression model to predict house prices using key features such as area, number of bedrooms, and number of bathrooms. It demonstrates a complete regression workflow from data loading to model evaluation.

Key Steps Covered
- Imported and inspected the housing dataset
- Selected features and target variable (price)
- Prepared the dataset for regression modeling
- Split data into training and testing sets
- Trained a Linear Regression model
- Generated predictions on the test set
- Evaluated model performance using:
    - Mean Squared Error (MSE)
    - R-squared score (R²)

Outcome

This notebook highlights how Linear Regression can be used for predicting continuous values such as house prices. It serves as a foundational example for exploring more advanced regression techniques in future notebooks.


File: 3. Iris_Dataset_Simple_Neural_Network.ipynb

This notebook builds a simple Neural Network classifier using TensorFlow/Keras to classify the Iris dataset into its three species. It demonstrates the full deep-learning workflow, including preprocessing, one-hot encoding, neural network design, training, and evaluation.

Key Steps Covered
- Loaded the Iris dataset using sklearn.datasets
- Converted the dataset into a Pandas DataFrame for inspection
- Prepared the feature matrix (X) and labels (y)
- Standardized the input features using StandardScaler
- One-hot encoded the target labels using OneHotEncoder
- Split the dataset into training and testing sets
- Built a simple feed-forward neural network using Keras:
    - Input layer
    - Hidden layer with ReLU activation
    - Output layer with softmax activation
- Compiled the model using:
    - Optimizer: Adam
    - Loss: Categorical Crossentropy
    - Metric: Accuracy
- Trained the neural network and reviewed training performance

Outcome

This notebook demonstrates the fundamental steps for creating a neural network classifier using TensorFlow/Keras. It provides a clear introduction to deep learning concepts such as activation functions, feature scaling, one-hot encoding, and model evaluation.


File: 4. Titanic-Neural_Network.ipynb

This notebook builds a Neural Network classifier to predict Titanic passenger survival using TensorFlow/Keras. It extends the earlier Decision Tree approach by applying a deep-learning model for binary classification.

Key Steps Covered
- Loaded the Titanic dataset from a public URL
- Dropped irrelevant or non-useful columns (PassengerId, Name, Ticket, Cabin)
- Filled missing values for Age and Embarked
- Converted categorical variables (Sex, Embarked) using one-hot encoding
- Defined features (X) and target variable (y)
- Split data into training and testing sets
- Scaled numerical features using StandardScaler
- Built a neural network with:
   - Two dense hidden layers (ReLU activation)
   - One output layer (sigmoid activation)
- Compiled the model using binary crossentropy loss and Adam optimizer
- Trained the network with validation split
- Evaluated model performance using test loss and accuracy

Outcome

This notebook demonstrates how to apply a deep-learning approach to a classic binary classification problem. It provides a clear introduction to building, training, and evaluating neural networks using Keras, expanding beyond traditional machine learning methods.
