import streamlit as st
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import streamlit as st
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Lasso, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression
from sklearn import linear_model
from sklearn.linear_model import RidgeCV

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold,StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn import datasets

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

st.set_page_config(layout="wide")


class DataPreprocessing:
    def __init__(self, df_or_path):
        if isinstance(df_or_path, pd.DataFrame):
            self.df = df_or_path
        elif isinstance(df_or_path, str):
            self.df = pd.read_csv(df_or_path, header=None)
        else:
            raise ValueError("Input must be a DataFrame or a file path.")

    def read_data(self):
        x = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]
        return x, y

    def split(self, x, y, split_size=80):
        stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=(100 - split_size) / 100, random_state=42)
        train_index, test_index = next(stratified_splitter.split(x, y))
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        return X_train, X_test, y_train, y_test

    def standardization(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled)
        X_test_scaled = pd.DataFrame(X_test_scaled)

        st.write("X_train_scaled:")
        st.write(X_train_scaled)
        st.write("X_test_scaled:")
        st.write(X_test_scaled)

        return X_train_scaled, X_test_scaled

    @classmethod
    def discretize(cls, x, n_bins=5, strategy='uniform'):
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        discretized_data = discretizer.fit_transform(x)
        discretized_df = pd.DataFrame(discretized_data, columns=x.columns)
        return discretized_df

    def show_data(self, X_train, X_test, x, y):
        st.markdown('**1.2. Data splits**')
        st.write('Training set')
        st.info(X_train.shape)
        st.write('Test set')
        st.info(X_test.shape)

        st.markdown('**1.3. Features/Variable Details**:')
        st.write('X variable')
        st.info(list(x.columns))
        st.write('Y variable')
        st.info(y.name)

        
# Page 1: Data Upload and Preprocessing
def page_data_preprocessing():
    st.write("# Page 1: Data Upload and Preprocessing")

    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("## View of dataset")
        st.write(df.head())
        #DATA PREPROCESSING
        new_obj = DataPreprocessing(df)
        x, y = new_obj.read_data()
        X_train, X_test, y_train, y_test = new_obj.split(x, y)
        # new_obj.cross_validate()  # Add parentheses to invoke the method
        new_obj.show_data(X_train, X_test, x, y)  # Add parentheses to invoke the method
        X_train_scaled, X_test_scaled = new_obj.standardization(X_train, X_test)
        # Store data in session state
        st.session_state.data_processed = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
        }

    else:
        st.info('Awaiting for CSV file to be uploaded.')
    # if st.button('Press to use Example Dataset'):
    #     # Boston housing dataset
    #     boston = datasets.load_boston()
    #     # boston = load_boston()
    #     X = pd.DataFrame(boston.data, columns=boston.feature_names)
    #     Y = pd.Series(boston.target, name='response')
    #     df = pd.concat( [X,Y], axis=1 )
    #     st.markdown('The Boston housing dataset is used as the example.')
    #     st.write(df.head(5))
        # Your data preprocessing logic goes here


# Sidebar - Specify parameter settings
#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    #uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    """)

with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])





class Regularizer:
    def __init__(self, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled

    def lasso(self):
        lasso_cv = LassoCV(alphas=[0.1, 0.2, 0.3], cv=10, max_iter=11000)
        lasso_cv.fit(self.X_train, self.y_train)

        best_alpha = lasso_cv.alpha_
        st.write(f"Best alpha (Lasso): {best_alpha}")

        lasso_model = Lasso(alpha=best_alpha)
        lasso_model.fit(self.X_train_scaled, self.y_train)
        coefficients = lasso_model.coef_

        X_train_lasso = self.X_train_scaled * coefficients
        X_test_lasso = self.X_test_scaled * coefficients

        st.write("X_train_lasso:")
        st.write(X_train_lasso)
        st.write("X_test_lasso:")
        st.write(X_test_lasso)

        return X_train_lasso, X_test_lasso, coefficients

    def ridge(self):
        ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=10)
        ridge_cv.fit(self.X_train_scaled, self.y_train)

        ridge_best_alpha = ridge_cv.alpha_
        st.write(f"Best alpha (Ridge): {ridge_best_alpha}")

        ridge_model = Ridge(alpha=ridge_best_alpha)
        ridge_model.fit(self.X_train_scaled, self.y_train)
        coefficients = ridge_model.coef_

        X_train_ridge = self.X_train_scaled * coefficients
        X_test_ridge = self.X_test_scaled * coefficients

        st.write("X_train_ridge:")
        st.write(X_train_ridge)
        st.write("X_test_ridge:")
        st.write(X_test_ridge)

        return X_train_ridge, X_test_ridge

    def pls(self, n_components=7):
        pls_model = PLSRegression(n_components=n_components)
        pls_model.fit(self.X_train_scaled, self.y_train)

        X_train_pls = pls_model.transform(self.X_train_scaled)
        X_test_pls = pls_model.transform(self.X_test_scaled)

        imputer = SimpleImputer(strategy='mean')
        X_train_pls_imputed = imputer.fit_transform(X_train_pls)
        X_test_pls_imputed = imputer.transform(X_test_pls)

        train_feature_pls = pd.DataFrame(X_train_pls_imputed)
        test_feature_pls = pd.DataFrame(X_test_pls_imputed)

        st.write("train_feature_pls:")
        st.write(train_feature_pls)
        st.write("test_feature_pls:")
        st.write(test_feature_pls)
        return train_feature_pls, test_feature_pls

    @classmethod
    def discretize(cls, x, n_bins=5, strategy='uniform'):
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        discretized_data = discretizer.fit_transform(x)
        discretized_df = pd.DataFrame(discretized_data, columns=x.columns)
        return discretized_df

    def mutual_information(self, components=7):
        X_train_discretized = self.discretize(self.X_train_scaled)
        X_test_discretized = self.discretize(self.X_test_scaled)

        mi_scores = mutual_info_classif(X_train_discretized, self.y_train)
        mi_series = pd.Series(mi_scores, index=X_train_discretized.columns)
        mi_sorted = mi_series.sort_values(ascending=False)
        selected_features = mi_sorted.head(components).index

        X_selected_train = self.X_train[selected_features]
        X_selected_test = self.X_test[selected_features]

        st.write("Selected features based on Mutual Information:")
        st.write(X_selected_train)
        st.write("Selected features based on Mutual Information:")
        st.write(X_selected_test)
        return X_selected_train, X_selected_test




class Classifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    def cross_validate_models(self, X, y, models, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)):
        scores_dict = {}

        for model_name, model in models:
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            scores_dict[model_name] = scores

        scores_df = pd.DataFrame(scores_dict)
        return scores_df
    def plot_radar(self, results_df, title_suffix=""):
        categories = list(results_df.columns)
        models = results_df.index
        num_models = len(models)

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values = results_df.values.T

        values = np.concatenate((values, [values[:, 0]]), axis=1)
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        for i in range(num_models):
            ax.plot(angles, values[:, i], label=models[i])

        ax.fill(angles, values.mean(axis=1), color='red', alpha=0.25)
        ax.set_yticklabels([])
        plt.title('Radar Plot - {}'.format(title_suffix))
        plt.legend(loc="upper right")
        plt.show()

    def plot_roc_auc(self, model, probabilities, title_suffix=""):
        roc_auc = roc_auc_score(self.y_test, probabilities)
        # ROC curve and AUC plot
        fpr, tpr, thresholds = roc_curve(self.y_test, probabilities)
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - {}'.format(title_suffix))
        plt.legend(loc="lower right")
        plt.show()

        return roc_auc

    def plot_scatter(self, results_dict, title_suffix=""):
        categories = list(results_dict.keys())
        values = list(results_dict.values())
        plt.figure(figsize=(8, 6))
        plt.scatter(categories, values, color='blue')
        plt.title('Scatter Plot - {}'.format(title_suffix))
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.show()

    def knn(self):
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(self.X_train, self.y_train)
        predictions_knn = knn_classifier.predict(self.X_test)
        probabilities_knn = knn_classifier.predict_proba(self.X_test)[:, 1]

        precision = precision_score(self.y_test, predictions_knn)
        recall = recall_score(self.y_test, predictions_knn)
        accuracy = accuracy_score(self.y_test, predictions_knn)
        f1_measure = f1_score(self.y_test, predictions_knn)
        roc_auc = self.plot_roc_auc(knn_classifier, probabilities_knn, title_suffix="KNN Classification")
        result_dict = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'f-score': f1_measure, 'ROC-AUC': roc_auc}
        return result_dict

    def svm(self):
        svm_classifier = SVC(probability=True)
        svm_classifier.fit(self.X_train, self.y_train)
        predictions_svm = svm_classifier.predict(self.X_test)
        probabilities_svm = svm_classifier.predict_proba(self.X_test)[:, 1]

        precision = precision_score(self.y_test, predictions_svm)
        recall = recall_score(self.y_test, predictions_svm)
        accuracy = accuracy_score(self.y_test, predictions_svm)
        f1_measure = f1_score(self.y_test, predictions_svm)
        roc_auc = self.plot_roc_auc(svm_classifier, probabilities_svm, title_suffix="SVM Classification")
        result_dict = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'f-score': f1_measure, 'ROC-AUC': roc_auc}
        return result_dict

    def lda(self):
        lda_model = LinearDiscriminantAnalysis()
        lda_model.fit(self.X_train, self.y_train)
        predictions_lda = lda_model.predict(self.X_test)
        probabilities_lda = lda_model.predict_proba(self.X_test)[:, 1]

        precision = precision_score(self.y_test, predictions_lda)
        recall = recall_score(self.y_test, predictions_lda)
        accuracy = accuracy_score(self.y_test, predictions_lda)
        f1_measure = f1_score(self.y_test, predictions_lda)
        roc_auc = self.plot_roc_auc(lda_model, probabilities_lda, title_suffix="LDA Classification")

        result_dict = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'f-score': f1_measure, 'ROC-AUC': roc_auc}
        return result_dict

    def decision_tree(self):
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(self.X_train, self.y_train)
        predictions_decision_tree = decision_tree.predict(self.X_test)
        probabilities_decision_tree = decision_tree.predict_proba(self.X_test)[:, 1]

        precision = precision_score(self.y_test, predictions_decision_tree)
        recall = recall_score(self.y_test, predictions_decision_tree)
        accuracy = accuracy_score(self.y_test, predictions_decision_tree)
        f1_measure = f1_score(self.y_test, predictions_decision_tree)
        roc_auc = self.plot_roc_auc(decision_tree, probabilities_decision_tree, title_suffix="Decision Tree Classification")
        result_dict = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'f-score': f1_measure, 'ROC-AUC': roc_auc}
        return result_dict




# Page 2: Run Regularizers
def page_run_regularizers():
    st.write("# Page 2: Run Regularizers")

    # Retrieve session state
    if 'data_processed' not in st.session_state:
        st.warning("Please upload and preprocess data on Page 1.")
        return

    # Accessing data from the session state
    X_train = st.session_state.data_processed['X_train']
    X_test = st.session_state.data_processed['X_test']
    y_train = st.session_state.data_processed['y_train']
    y_test = st.session_state.data_processed['y_test']
    X_train_scaled = st.session_state.data_processed['X_train_scaled']
    X_test_scaled = st.session_state.data_processed['X_test_scaled']

    # Regularizer option
    regularizer_option = st.selectbox('Select Regularizer', ['Lasso', 'Ridge', 'PLS','Mutual Information'])

    if regularizer_option == 'Lasso':
        st.write("## Running Lasso Regularizer")
        lasso_model = Regularizer(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)  
        train_feature_lasso, test_feature_lasso, coefficients = lasso_model.lasso()
           
        # Total number of non-zero columns
        total_non_zero_train = (coefficients != 0).sum()

        st.write("\nTotal number of non-zero columns for training data:", total_non_zero_train)
        components = total_non_zero_train
        st.write("Number of components:", components)

        # Display Lasso features
        lasso_features=train_feature_lasso.iloc[:,coefficients!=0]
        st.write("Lasso Features:")
        st.write(lasso_features)

        st.markdown('**3.2. Regularization**')
        st.write('LASSO Regularized Dataset')
        st.info(train_feature_lasso.shape)
        st.write('Test set')
        st.info(test_feature_lasso.shape)

        st.markdown('**3.3. Lasso Features/Variable Details**:')
        st.write('Lasso Regularized Variables')
        st.info(list(lasso_features.columns))
        # st.write('Y variable')
        # st.info(y.name)
        # # Perform further operations...
        # clf = Classifier(train_feature_lasso,test_feature_lasso,y_train, y_test)

        # models_to_cross_validate = [
        #     ('KNN', KNeighborsClassifier(n_neighbors=3)),
        #     ('SVM', SVC(probability=True)),
        #     ('LDA', LinearDiscriminantAnalysis()),
        #     ('Decision Tree', DecisionTreeClassifier())
        # ]
        # cross_val_scores = clf.cross_validate_models(train_feature_lasso, y_train, models_to_cross_validate)
        
        # # # Display Cross-Validation Box Plot
        # boxplot_figure, boxplot_ax = plt.subplots()
        # boxplot_ax.boxplot(cross_val_scores.values, labels=cross_val_scores.keys())
        # boxplot_ax.set_title('Cross-Validation Box Plot using Lasso Regression')
        # boxplot_ax.set_ylabel('Accuracy')
        # st.pyplot(boxplot_figure)

        # # ROC-AUC plot for each model
        # for model_name, model in models_to_cross_validate:
        #     roc_auc_figure, roc_auc_ax = plt.subplots(figsize=(8, 8))
        #     probabilities = model.fit(clf.X_train, clf.y_train).predict_proba(clf.X_test)[:, 1]
        #     clf.plot_roc_auc(model, probabilities, title_suffix=f'{model_name} - ROC-AUC')
        #     roc_auc_ax.set_title(f'{model_name} - ROC-AUC')
        #     st.pyplot(roc_auc_figure)
        # # Perform further operations as needed...

    elif regularizer_option == 'Ridge':
        st.write("## Running Ridge Regularizer")
        ridge_model = Regularizer(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
        train_feature_ridge, test_feature_ridge = ridge_model.ridge()

        # Perform further operations as needed...

    elif regularizer_option == 'PLS':
        st.write("## Running PLS Regularizer")
        pls_model = Regularizer(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
        train_feature_pls, test_feature_pls = pls_model.pls()
        
    elif regularizer_option == 'Mutual Information':
        st.write("## Running MI Regularizer")
        mi_model = Regularizer(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
        X_selected_train, X_selected_test = mi_model.mutual_information(4)
        
    # clf = Classifier(train_feature_pls, test_feature_pls, y_train, y_test)

    # models_to_cross_validate = [
    #     ('KNN', KNeighborsClassifier(n_neighbors=3)),
    #     ('SVM', SVC(probability=True)),
    #     ('LDA', LinearDiscriminantAnalysis()),
    #     ('Decision Tree', DecisionTreeClassifier())
    # ]

    # # Perform cross-validation and get scores
    # cross_val_scores = clf.cross_validate_models(train_feature_pls, y_train, models_to_cross_validate)

    # # Create a box plot of the cross-validation scores
    # st.pyplot(plt.boxplot(cross_val_scores.values, labels=cross_val_scores.keys()))
    # st.title('Cross-Validation Box Plot using PLS Regression')
    # st.ylabel('Accuracy')

    # # ROC-AUC plot for each model
    # for model_name, model in models_to_cross_validate:
    #     probabilities = model.fit(clf.X_train, clf.y_train).predict_proba(clf.X_test)[:, 1]
    #     st.pyplot(plt.figure(figsize=(8, 8)))
    #     clf.plot_roc_auc(model, probabilities, title_suffix=f'{model_name} - ROC-AUC')
    #     st.title(f'{model_name} - ROC-AUC')



# Main function
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Preprocessing", "Run Regularizers"])
   
    if page == "Data Preprocessing":
        page_data_preprocessing()
        
    elif page == "Run Regularizers":
        page_run_regularizers()

if __name__ == "__main__":
    main()
