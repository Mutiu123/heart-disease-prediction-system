## **Project Description**:
Heart disease remains a significant global health concern, affecting millions of people each year. Early detection and accurate prediction of heart disease are crucial for effective prevention and timely intervention. In this project, we aim to develop a heart disease prediction system using machine learning techniques. By analysing patient data and identifying relevant risk factors, we can assist healthcare professionals in making informed decisions and improving patient outcomes.

   - The project aims to predict heart disease based on various features using a logistic regression model.
   - The system takes input data related to patients (such as age, gender, smoking status, cholesterol levels, etc.) and predicts whether the patient has heart disease or not.
   - The project is implemented in Python using libraries like NumPy, pandas, and scikit-learn.
   - Hyperparameter tuning is performed using GridSearchCV to find the best value of the regularization parameter C.

## **Problem Statement**:
The primary objective is to build a reliable predictive model that can assess the likelihood of heart disease based on patient characteristics. The problem involves binary classification: given a set of features (such as age, gender, cholesterol levels, smoking status, and blood pressure), I need to predict whether a patient is at risk of heart disease or not. The challenge lies in balancing accuracy, sensitivity (identifying true positive cases), and specificity (minimising false positives) to create a robust model.

## **Methodology**:

1. **Data Preprocessing:**
   - **Data Loading**: We begin by loading the heart disease dataset. This dataset likely contains information about patients, including features like age, gender, cholesterol levels, and blood pressure.
   - **Handling Missing Values**: Missing data can adversely affect model performance. We address this by identifying missing values and deciding how to handle them (e.g., imputation or removal).
   - **Encoding Categorical Variables**: Since machine learning models require numerical input, we convert categorical variables (such as gender) into numerical representations (e.g., 0 for female and 1 for male).
   - **Data Quality Assurance**: Ensuring data quality is crucial. We validate that the dataset is clean, consistent, and free from anomalies.

2. **Feature Selection and Engineering:**
   - **Feature Analysis**: We explore the dataset to understand which features are relevant for predicting heart disease. Some features may have stronger associations with the target variable (heart disease) than others.
   - **Feature Engineering**: We create new features or transform existing ones to enhance model performance. For instance, we might calculate the body mass index (BMI) from height and weight or derive a composite risk score based on multiple features.
   - **Dimensionality Reduction (if needed)**: If the dataset has many features, we consider techniques like Principal Component Analysis (PCA) to reduce dimensionality while preserving important information.

3. **Model Selection and Hyperparameter Tuning:**
   - **Choosing the Model**: Logistic regression is a suitable choice due to its simplicity and interpretability. However, we could explore other models (e.g., decision trees, random forests, or support vector machines) to compare their performance.
   - **Hyperparameter Tuning**: We use techniques like cross-validation and `GridSearchCV` to find optimal hyperparameters. For logistic regression, the regularization parameter `C` is crucial. We search for the best `C` value that balances bias and variance.
   - **Model Training**: Once we determine the best hyperparameters, we train the logistic regression model on the training data.

4. **Evaluation Metrics:**
   - We assess the model's performance using various metrics:
     - **Accuracy**: Overall correctness of predictions.
     - **Precision**: Proportion of true positive predictions among all positive predictions.
     - **Recall (Sensitivity)**: Proportion of true positive predictions among actual positive cases.
     - **F1-score**: Harmonic mean of precision and recall.
     - **Area Under the Receiver Operating Characteristic (ROC-AUC)**: Measures the model's ability to distinguish between positive and negative cases.


## **Contributions**:

1. **Data Preprocessing and Quality Assurance**:
   - **Handling Missing Values**: My contribution involves addressing missing data within the heart disease dataset. By using `data.dropna(inplace=True)`, you ensure that incomplete records do not adversely affect model training.
   - **Data Cleaning**: Removing unnecessary columns (e.g., 'education') improves data quality. Clean data is essential for reliable predictions.

2. **Feature Selection and Engineering**:
   - **Feature Analysis**: I explore the dataset to identify relevant features. This step is crucial because not all variables contribute equally to predicting heart disease.
   - **Feature Engineering**: My work extends beyond raw features. I may create new variables (e.g., BMI) or transform existing ones (e.g., risk scores) to enhance the model's performance. Feature engineering is an art that requires domain knowledge and creativity.

3. **Model Selection and Hyperparameter Tuning**:
   - **Choosing the Model**: Logistic regression is a sensible choice due to its interpretability. My decision aligns with the project's goal of creating an understandable model for healthcare professionals.
   - **Hyperparameter Tuning**: By using `GridSearchCV`, I systematically explore different hyperparameter values (e.g., regularization strength). Fine-tuning ensures optimal model performance.

4. **Model Evaluation and Metrics**:
   - **Accuracy Assessment**: I evaluate the model's accuracy on both the training and test sets. This step provides insights into how well the model generalizes to unseen data.
   - **Precision, Recall, and F1-score**: Beyond accuracy, precision (true positive rate), recall (sensitivity), and F1-score (harmonic mean) help assess the model's performance across different aspects of prediction.

## **Impact and Practical Application**:
   - My contributions extend beyond code. By building this heart disease prediction system, you empower healthcare professionals to make informed decisions. Early detection can lead to timely interventions, potentially saving lives.
   - Patient Empowerment: Patients can also benefit from this system. Awareness of their risk factors allows them to take proactive steps toward heart health, such as lifestyle modifications or seeking medical advice.
   - Educational Value: My work serves as an educational resource, demonstrating the application of machine learning in healthcare. It contributes to knowledge dissemination and fosters understanding among practitioners and students.

## **Code Breakdown and Explanations**:

1. **Data Preprocessing:**
   - **Data Loading**: We load the heart disease dataset using `pd.read_csv("data/heart_disease.csv")`. This step ensures that we have access to the necessary patient data.
   - **Handling Missing Values**: The `data.dropna(inplace=True)` line removes rows with missing values. Handling missing data is crucial for accurate predictions.
   - **Encoding Categorical Variables**: We convert categorical variables ('Gender', 'Heart_ stroke', 'prevalentStroke') into numerical representations (0 or 1) using `replace()`. This transformation allows us to use these features in our model.

2. **Data Splitting**:
   - We split the data into training and test sets using `train_test_split(inputData, outputData, test_size=0.25)`. The training set is used to train the model, while the test set evaluates its performance.

3. **Model Building and Hyperparameter Tuning**:
   - We choose logistic regression (`LogisticRegression()`) as our predictive model. Its simplicity and interpretability make it suitable for this task.
   - Hyperparameter tuning is essential. We use `GridSearchCV` to search for the best value of the regularization parameter `C`. The `param_grid` specifies a range of `C` values to explore.

4. **Model Evaluation**:
   - After training the model with the best `C`, we evaluate its performance on both the training and test sets.
   - The `model.score(x_test, y_test)` line provides the accuracy score on the test data.



## **System Demo:**

![The System Demo](https://github.com/Mutiu123/heart-disease-prediction-system/blob/main/demos/demo1.png)

![The System Demo](https://github.com/Mutiu123/heart-disease-prediction-system/blob/main/demos/demo2.png)


## **To run the model**
1. **Clone the Repository**:
   - First, clone the repository containing the heart disease prediction system code to your local machine. You can do this using Git or by downloading the ZIP file from the repository.

2. **Install Dependencies**:
   - Open your terminal or command prompt and navigate to the project directory.
   - Install the necessary Python libraries mentioned in the `requirements.txt` file using the following command:
     ```
     pip install -r requirements.txt
     ```

3. **Run the Streamlit App**:
   - In the same terminal or command prompt, execute the following command to run the Streamlit app:
     ```
     streamlit run app.py
     ```
   - This will start the local development server, and you'll see a message indicating that the app is running.
   - Open your web browser and visit `http://localhost:8501` (or the URL provided in the terminal) to access the interactive web app.

4. **Predict Heart Disease**:
   - On the Streamlit app, you'll find a search bar where you can Fill in the relevant information for a patient you want to assess.
   - After entering the patient’s details, click the “Predict” button.

## **Model Deployement**
I Deploy the Streamlit app on Heroku to allows others to access it online Here's an updated step-by-step guide on how to run the app on your device:

1. **Access the Deployed App**:
   - Visit the following link: [Heart Disease Prediction System](https://heart-disease-prediction-syste-c5ca4dc4ba23.herokuapp.com/).
   - You'll see the web interface where users can input patient information and get predictions.

2. **Interact with the App**:
   - On the web page, you'll find input fields for patient details such as age, gender, smoking status, cholesterol levels, and more.
   - Fill in the relevant information for a patient you want to assess.

3. **Click the "Predict" Button**:
   - After entering the patient's details, click the "Predict" button.
   - The app will process the input using the trained logistic regression model.

4. **View the Prediction**:
   - The app will display the prediction result:
     - If the output is 0, it means "Patient