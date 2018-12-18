### N-ary Decision Tree classifier
Chooses the split along lines with the highest information gain as measured by two metrics: entropy and gini impurity.

* decision_tree.py contains the code for fit and predict.

* DecisionTreeExecution.ipynb imports decision_tree classifier and tests it against zoo animals dataset.

#### Concept:

The root node will receive the entire training set.
Each node will ask a true / false question about one of the features.
In response to this question, we split the data into two subsets which are the inputs to the child nodes we add to the tree.

The goal is to unmix the labels at each node. In other words, our goal is to produce the purest possible distribution of labels at each node.

Understand which questions to ask and when.
Gini impurity quantifies how much uncertainity there is at a node. (If the right label is marked, the impurity is 0. If the label is mixed up, the impurity is > 0)
Information gain quantifies how much a question reduces the uncertainity at a node.
