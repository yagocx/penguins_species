# Penguin Species Classification with Decision Tree

This project demonstrates a complete pipeline for preprocessing, training, and evaluating a decision tree model applied to the Penguin dataset (available via Seaborn). The code also includes visualization of the decision tree and confusion matrix, as well as a test of the model with a new input.

## Project Overview

The script performs the following steps:

1. **Data Loading:**  
   Uses the "penguins" dataset from the Seaborn library.

2. **Missing Data Handling:**  
   - For categorical columns (`species`, `island`, and `sex`), rows with missing values are dropped.  
   - For numerical columns, missing values are replaced with the median.

3. **Feature Standardization:**  
   - Numerical variables are standardized (mean = 0, standard deviation = 1) for: `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, and `body_mass_g`.
   - Categorical variables are transformed into dummy variables (0 or 1) for each value of `island` and for the `sex` column.

4. **Data Cleaning:**  
   Selects only the relevant features (input features and target `species` column).

5. **Train/Test Split:**  
   Splits the data using `train_test_split`, reserving 1/3 for testing.

6. **Model Training:**  
   Trains a decision tree classifier using scikit-learn's `DecisionTreeClassifier`.

7. **Tree Visualization:**  
   Displays the decision tree using `plot_tree` with Matplotlib.

8. **Model Evaluation:**  
   - Confusion matrix is computed with `confusion_matrix` and visualized using `ConfusionMatrixDisplay`.
   - Model accuracy is computed using `accuracy_score`.
   - The accuracy score is printed to the console.

9. **Model Test with New Example:**  
   A standardized penguin sample is created and the model predicts its species.

## Requirements

This project requires the following Python libraries:

- **Python 3.x**
- **NumPy** – for numerical operations
- **Pandas** – for data manipulation
- **Seaborn** – for dataset and visualization
- **Matplotlib** – for plotting
- **scikit-learn** – for model building and evaluation

## Example Test Case
The model is tested with a synthetic penguin having these characteristics (standardized based on the original dataset):

Bill Length: 38.2 mm

Bill Depth: 18.1 mm

Flipper Length: 185.0 mm

Body Mass: 3950.0 g

Island: Biscoe (represented as a dummy variable)

Sex: Male (represented as a dummy variable)

- The predicted species is printed to the console.

## Notes
The trained decision tree has 8 leaves.

The confusion matrix shows good overall performance, although some misclassifications occur between 'Adelie' and 'Chinstrap', as well as between 'Chinstrap' and 'Gentoo'.

Further improvements might be achieved by enhancing feature selection or exploring more advanced models.

## License

This project is licensed under the MIT License.

## Acknowledgements

This project uses the penguins dataset provided by seaborn.  
Happy coding!

---

