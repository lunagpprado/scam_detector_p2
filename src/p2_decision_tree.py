# # Scam Detector: Random Forest

#     This Jupyter Notebook will be used to run a Random Forest Algorithm to predict if a given email is a scam or a ham(a normal email).

# ## Import Packages


#import the packages we need
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score # Used for accuracy calculation
from sklearn.decomposition import TruncatedSVD # Used for dimensionality reduction
from sklearn.feature_extraction.text import TfidfVectorizer # Used for text numerical vectorization
from sklearn.model_selection import train_test_split # Used for randomly splitting between training and validation data
from sklearn.preprocessing import StandardScaler # Used for standardizing vector distributions


# ## Retrieve Data


col_names = ['sender', 'receiver', 'subject', 'body', 'label', 'urls']
path = "./data/CEAS_08.csv"
data1 = pd.read_csv(path)
data1 = data1.drop('date', axis=1) # Data wouldn't affect whether something is a scam or not
col_names[-1], col_names[-2] = col_names[-2], col_names[-1]
data1 = data1[col_names]
data1.head(10)
counts = data1['urls'].value_counts() # Used for checking how even different features are represented in the data.

data1 = data1.drop('receiver', axis=1) # Reciever wouldn't affect whether something is a scam or not

data = data1

# Processes all na values with Empty strings
data['sender'].fillna(' ', inplace=True)
data['subject'].fillna(' ', inplace=True)
data['body'].fillna(' ', inplace=True)
#data['sender'].isnull().any() # used to check if any nulls remain


# ## TF-IDF and Data Processing


# Text Vectorization for numeric representation of the text data
corpus_sender = data['sender'][0:]
vectorizer_send = TfidfVectorizer()
send = vectorizer_send.fit_transform(corpus_sender)
svd_send = TruncatedSVD(n_components=5, random_state=42) # Performs SVD to reduce the dimensionality of the vectors into something usable.
send_reduced = svd_send.fit_transform(send)

# Text Vectorization for numeric interpretations
corpus_sub = data['subject'][0:]
vectorizer_sub = TfidfVectorizer()
sub = vectorizer_sub.fit_transform(corpus_sub)

svd_sub = TruncatedSVD(n_components=50, random_state=42)
sub_reduced = svd_sub.fit_transform(sub)

# Text Vectorization for numeric interpretations
corpus_body = data['body'][0:]
vectorizer_body = TfidfVectorizer()
body = vectorizer_body.fit_transform(corpus_body)

svd = TruncatedSVD(n_components=300, random_state=42) 
body_reduced = svd.fit_transform(body)

dupes_data = data.index[data.index.duplicated()] # Checks for duplicate rows
#print("data duplicates:", dupes_data)

scaler_body = StandardScaler() # Performs a standard scaler z=(x-mu)/sigma, in order to get it to a mean of 0 and variance of 1 for easier algorithmic analysis
body_reduced = scaler_body.fit_transform(body_reduced)

scaler_send = StandardScaler()
send_reduced = scaler_send.fit_transform(send_reduced)

scaler_sub = StandardScaler()
sub_reduced = scaler_sub.fit_transform(sub_reduced)

body_df = pd.DataFrame(body_reduced, columns=range(1,301))
send_df = pd.DataFrame(send_reduced, columns=range(1,6))
sub_df = pd.DataFrame(sub_reduced, columns=range(1,51))

new_data = pd.concat([body_df, send_df, sub_df, data], axis=1)
new_data = new_data.drop('body', axis=1)
new_data = new_data.drop('sender', axis=1)
new_data = new_data.drop('subject', axis=1)

data = new_data


# ## Decision Class


class Decision:
    """ A decision is used to ask the question at a decision node to split the data.
    This class records column number and values and matches the stored feature value to a give feature value
    """
    
    def __init__(self, feature_index, threshold):
        self.feature_index = feature_index
        self.threshold = threshold
        
    def ask(self, input):
        # Compares input feature value to stored value
        feature_val = input[self.feature_index]
        if isinstance(feature_val, (int, float, np.number)):
            return feature_val >= self.threshold
        else:
            return feature_val == self.threshold
        

# ## Helper Functions for Splitting


def divide_df(rows, decision):
    # Partitions a data frame
    # Check if each row matches decision, divide into true and false
    col = rows[:, decision.feature_index]
    if np.issubdtype(col.dtype, np.number):
        mask = col >= decision.threshold
    else:
        mask = col == decision.threshold
    left, right = rows[mask],rows[~mask]
    return left, right


def label_count(rows):
    # Counts the number of each classification in data frame
    y = rows[:, -1]
    unique, label_counts = np.unique(y, return_counts=True)
    return dict(zip(unique,label_counts))


def gini_impurity(rows):
    #Calculates Gini Impurity for a data frame of rows.
    y = rows[:, -1]
    _, label_counts = np.unique(y, return_counts=True)
    probs = label_counts/label_counts.sum()
    return 1.0 - np.sum(probs**2) # gini impurity formula of a data frame based on label 


def info_gain(left, right, curr_gini):
    #Information gain: Gini of the root node subtracted by the impurty of the two children nodes.
    if len(left) + len(right) == 0:
        return 0
    prob = float(len(left) / (len(left) + len(right)))
    return curr_gini - prob * gini_impurity(left) - (1 - prob) * gini_impurity(right) #Information gain formula


def threshold_candidates(col, max_thresh=5):
    #Choose candidate threshold split
    unique = np.unique(col)
    if len(unique) > max_thresh:
        quantile = np.linspace(0, 100, max_thresh + 2)[1:-1]
        unique = np.percentile(unique, quantile)
    if len(unique) > 1:
        return (unique[:-1] + unique[1:])/2
    else:
        return unique


def info_gain_split(rows):
    #Find best decision to make based on informaiton gain
    X = rows[:, :-1]
    y = rows[:, -1]
    curr_gini = gini_impurity(rows)
    feature_count = X.shape[1]
    
    highest_gain = 0
    optimal_decision = None
    
    for feature_index in range(feature_count):
        col = X[:, feature_index]
        
        #Candidate Thresholds
        thresholds = threshold_candidates(col) if np.issubdtype(col.dtype, np.number) else np.unique(col)
        
        for candidate in thresholds:
            if np.issubdtype(col.dtype, np.number):
                mask = col >= candidate #Mask is a true/false matrix which then gets the rows divide based on threshold
            else:
                mask = col == candidate
            
            if mask.sum() == 0 or mask.sum() == len(mask):
                continue
        
            left, right = rows[mask], rows[~mask]
            gain = info_gain(left, right, curr_gini)
            
            if gain > highest_gain:
                highest_gain, optimal_decision = gain, Decision(feature_index, candidate)
                
    return highest_gain, optimal_decision


# ## Build Tree and Node Classes


class LeafNode:
    # A leaf Node holdes classified data.
    # Holds a dictionary with class counts in the leaf.
    
    def __init__(self,rows):
        self.pred = label_count(rows)


class DecisionNode:
    # A Decision Node asks a Decision to be made.
    # Holds reference to a Decision, and two child nodes.
    
    def __init__(self, decision, left, right):
        self.decision = decision
        self.left = left
        self.right = right


def build_tree(rows, depth=0, max_depth=10, min_sample_split=2):
    # Recursively Builds tree. Fitting at the same time.
    if len(rows) < min_sample_split or depth >= max_depth:
        return LeafNode(rows)
    
    highest_gain, optimal_decision = info_gain_split(rows)
    
    #Base case no further gain
    if highest_gain < 1e-6 or optimal_decision is None:
        return LeafNode(rows)
    
    #Found Partition
    left, right = divide_df(rows, optimal_decision)
    
    #Recurse Left Subtree
    left_subtree = build_tree(left, depth+1, max_depth, min_sample_split)
    
    #Recurse Right Subtree
    right_subtree = build_tree(right, depth+1, max_depth, min_sample_split)
    
    #Return Decision Node
    return DecisionNode(optimal_decision, left_subtree, right_subtree)


def predict(row, curr_node):
    #Base Case: Curr node is a leaf
    if isinstance(curr_node, LeafNode):
        total = sum(curr_node.pred.values())
        return max(curr_node.pred, key=curr_node.pred.get), {k: v/total for k,v in curr_node.pred.items()}
    
    #Recurse the left or right subtree
    if curr_node.decision.ask(row):
        return predict(row, curr_node.left)
    else:
        return predict(row, curr_node.right)


# ## Random Forest:


class RandomForest:
    def __init__(self, tree_count=10, max_depth=10, min_sample_split=2, feature_count=None):
        self.tree_count = tree_count # number of trees in forest
        self.max_depth = max_depth # maximum depth(height) of each tree
        self.min_sample_split = min_sample_split # minimum sample split at each decision node
        self.feature_count = feature_count # number of features within the dataset
        self.trees = [] # list of tree pointers
        
    def fit(self, X, y):
        self.trees = []
        self.feature_count = X.shape[1]
        self.feature_subspaces = [] # feature subsets of each tree 
        
        for _ in range(self.tree_count):
            X_partial, y_partial = self.bootstrap(X, y) # selects a subset of data points
            feature_index = np.random.choice(self.feature_count, int(np.sqrt(self.feature_count)), replace=False) # selects a sqrt number of features to be used in each tree
            self.feature_subspaces.append(feature_index)
            X_subspace = X_partial[:, feature_index] # indexes the feature subset
            rows = np.concatenate((X_subspace, y_partial), axis=1)
            
            tree = build_tree(rows, max_depth=self.max_depth, min_sample_split=self.min_sample_split) # builds each subsetted tree
            self.trees.append(tree)
        
        
    def bootstrap(self, X, y):
        # Selects a random sample of the data points for a tree
        sample_count = X.shape[0]
        row_index = np.random.choice(sample_count, sample_count, replace=True)
        return X[row_index], y[row_index]
        
    
    def subspace(self, X):
        # selects a random sample of features for a tree
        feature_index = np.random.choice(self.feature_count, int(self.feature_count**0.5), replace=False)
        return X[:, feature_index]
       
                                  
    def predict_one(self, X):
        # Returns one prediction from the trees based on a new input X
        votes = []
        vector = []
        for tree, features in zip(self.trees, self.feature_subspaces):
            X_subspace = X[features]
            pred, _  = predict(X_subspace, tree)
            vector.append(pred)
            votes.append(pred)
        return max(set(votes), key=votes.count), vector


# ## Prediction and Testing


def rf_predict(forest, sender, subject, body, url=0):
    input_data = {'sender': [sender], 'subject': [subject], 'body':[body], 'urls':[url]}
    input_df = pd.DataFrame(input_data)
    
    input_body = vectorizer_body.transform(input_df['body'])
    input_body_reduced = svd.transform(input_body)
    input_body_reduced = scaler_body.transform(input_body_reduced)
    input_body_df = pd.DataFrame(input_body_reduced, columns=range(1,301))

    input_send = vectorizer_send.transform(input_df['sender'])
    input_send_reduced = svd_send.transform(input_send)
    input_send_reduced = scaler_send.transform(input_send_reduced)
    input_send_df = pd.DataFrame(input_send_reduced, columns=range(1,6))

    input_sub = vectorizer_sub.transform(input_df['subject'])
    input_sub_reduced = svd_sub.transform(input_sub)
    input_sub_reduced = scaler_sub.transform(input_sub_reduced)
    input_sub_df = pd.DataFrame(input_sub_reduced, columns=range(1,51))
    

    input_data = pd.concat([input_body_df, input_send_df, input_sub_df, input_df], axis=1)
    input_data = input_data.drop('body', axis=1)
    input_data = input_data.drop('subject', axis=1)
    input_data = input_data.drop('sender', axis=1)
    
    input_df = input_data
    
    input_x = input_df.to_numpy()[0,:]
    pred, votes = my_forest.predict_one(input_x)
    return pred, votes.count(pred)/10


X = data.to_numpy()[:,:-1]
y = data.to_numpy()[:, -1].reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

my_forest = RandomForest(tree_count=10, max_depth=10, min_sample_split=2, feature_count=X_train.shape[1])
my_forest.fit(X_train,y_train)

arr = np.array([my_forest.predict_one(X)[0] for X in X_test])

accuracy = accuracy_score(y_test, arr)
#print(accuracy)


#Example Normal Email:
#test_data = {'sender': ['luna_prado@gmail.com'], 'subject': ['Advisor Help'], 'body':['Hello Dr. Athienitis, can you help me with choosing classes for the upcoming semester. Look forward to staying in contact.'], 'urls':[0]}
#test_df = pd.DataFrame(test_data)

#Example Scam Email:
#test_data = {'sender': ['amazon.asjfnakjsnfkanf@gmail.com'], 'subject': ['SCAM URGENT'], 'body':['Make money quick, urgent new opportunity. Please buy now for your future. Passive Income, Easy life. Venmo.com. Akjfaksjnkasfjna.com'], 'urls':[1]}
#test_df = pd.DataFrame(test_data)