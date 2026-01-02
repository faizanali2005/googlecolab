import math
from collections import defaultdict, Counter
# *** CORRECTED: Import all necessary types from typing ***
from typing import List, Dict, Any, Tuple, Optional

# ===================================================================
# 1. DUMMY DATA GENERATION (10 Customers)
# ===================================================================

# Features: Wait Time (Categorized), Food Temp (Categorized), Accuracy (0-100), Cleanliness (1-5), Prev Rating (1-5)
# Target: Satisfaction (1-5 Stars)

DUMMY_DATA = [
    # Wait Time, Food Temp, Accuracy, Cleanliness, Prev Rating, Satisfaction
    ("Fast", "Hot", 95, 5, 5, 5),  # Perfect Experience -> 5
    ("Avg", "Warm", 88, 4, 4, 4),  # Good -> 4
    ("Slow", "Hot", 98, 3, 3, 2),  # Great food, slow service -> 2
    ("Fast", "Hot", 100, 5, 5, 5),  # Perfect -> 5
    ("Avg", "Cold", 70, 2, 1, 1),  # Everything wrong -> 1
    ("Slow", "Warm", 60, 3, 2, 2),  # Slow and inaccurate -> 2
    ("Fast", "Hot", 99, 5, 5, 5),  # Perfect -> 5
    ("Avg", "Warm", 80, 4, 3, 3),  # Average everything -> 3
    ("Fast", "Hot", 92, 5, 4, 5),  # Very Good -> 5
    ("Slow", "Cold", 50, 1, 1, 1)  # Total failure -> 1
]

# Separate features (X) and target (Y)
FEATURES = [row[0:5] for row in DUMMY_DATA]
TARGETS = [row[5] for row in DUMMY_DATA]

# Map feature indices to names for clarity
FEATURE_NAMES = ["Wait_Time", "Food_Temp", "Accuracy", "Cleanliness", "Prev_Rating"]


# ===================================================================
# 2. NAIVE BAYES CLASSIFIER (From Scratch)
# ===================================================================

class NaiveBayesClassifier:
    def __init__(self):
        self.priors: Dict[int, float] = {}
        self.likelihoods: Dict[int, Dict[str, Dict[Any, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float)))
        self.class_feature_counts: Dict[int, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        self.class_totals: Counter = Counter()
        self.unique_classes: set = set()

    def train(self, X: List[Tuple], Y: List[int]):
        N = len(X)
        if N == 0:
            return

        # 1. Calculate Priors P(Class)
        class_counts = Counter(Y)
        self.unique_classes = set(Y)
        for cls, count in class_counts.items():
            self.priors[cls] = count / N
            self.class_totals[cls] = count

        # 2. Collect counts for Likelihoods
        for i, features in enumerate(X):
            cls = Y[i]

            for feature_index, feature_value in enumerate(features):
                feature_name = FEATURE_NAMES[feature_index]
                self.class_feature_counts[cls][feature_name][feature_value] += 1

        # Convert counts to probabilities (Likelihoods)
        self._calculate_likelihoods()

    def _calculate_likelihoods(self):
        # Calculate smoothed likelihoods using Laplace (Add-1) Smoothing

        all_feature_values = defaultdict(set)
        for cls in self.unique_classes:
            for feature_name in FEATURE_NAMES:
                for val in self.class_feature_counts[cls][feature_name]:
                    all_feature_values[feature_name].add(val)

        K = 1  # Smoothing constant

        for cls in self.unique_classes:
            total_in_class = self.class_totals[cls]

            for feature_index, feature_name in enumerate(FEATURE_NAMES):

                V = len(all_feature_values[feature_name]) if all_feature_values[feature_name] else 1
                denominator = total_in_class + V * K

                for feature_value in all_feature_values[feature_name]:
                    count = self.class_feature_counts[cls][feature_name].get(feature_value, 0)

                    likelihood = (count + K) / denominator

                    self.likelihoods[cls][feature_name][feature_value] = likelihood

    def predict(self, X_new: Tuple) -> int:
        new_features_list = [X_new]
        posteriors: Dict[int, float] = {}

        for features in new_features_list:
            for cls in self.unique_classes:
                log_posterior = math.log(self.priors.get(cls, 1e-10))

                for feature_index, feature_value in enumerate(features):
                    feature_name = FEATURE_NAMES[feature_index]

                    likelihood = self.likelihoods[cls][feature_name].get(
                        feature_value,
                        1e-10  # Fallback for unseen combinations
                    )

                    log_posterior += math.log(likelihood)

                posteriors[cls] = log_posterior

            predicted_class = max(posteriors, key=posteriors.get)
            return predicted_class

        return 3

    def explain_prediction(self, features: Tuple, prediction: int):
        print(f"\n--- NB Prediction Explanation for Input: {features} ---")
        print(f"Predicted Satisfaction: {prediction} Stars")

        log_posteriors: Dict[int, float] = {}

        for cls in self.unique_classes:
            log_posterior = math.log(self.priors.get(cls, 1e-10))

            for feature_index, feature_value in enumerate(features):
                feature_name = FEATURE_NAMES[feature_index]
                likelihood = self.likelihoods[cls][feature_name].get(feature_value, 1e-10)
                log_posterior += math.log(likelihood)

            log_posteriors[cls] = log_posterior

        print("\n[Prior Probabilities (P(Class))]:")
        for cls, p in sorted(self.priors.items(), key=lambda item: item[0]):
            print(f"  P({cls} Stars) = {p:.3f}")

        print("\n[Likelihoods Driving Prediction]:")
        for i, feature_name in enumerate(FEATURE_NAMES):
            val = features[i]
            likelihood_pred = self.likelihoods[prediction][feature_name].get(val, 1e-10)
            print(
                f"  Feature '{feature_name}'='{val}': P(F | {prediction} Stars) = {likelihood_pred:.4f} (Contribution: {math.log(likelihood_pred):.3f})")

        print(f"\n[Final Log Scores (Higher is better)]:")
        for cls, score in sorted(log_posteriors.items(), key=lambda item: item[1], reverse=True):
            print(f"  Score for {cls} Stars: {score:.3f}")


# ===================================================================
# 3. DECISION TREE (From Scratch)
# ===================================================================

# --- Entropy and Information Gain Calculation ---

def calculate_entropy(labels: List[int]) -> float:
    """Calculates Shannon Entropy for a set of class labels."""
    if not labels:
        return 0.0

    N = len(labels)
    counts = Counter(labels)
    entropy = 0.0

    for count in counts.values():
        probability = count / N
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy


def calculate_information_gain(parent_labels: List[int], subsets: List[List[int]]) -> float:
    """Calculates Information Gain from splitting the parent set into subsets."""

    parent_entropy = calculate_entropy(parent_labels)
    weighted_subset_entropy = 0.0
    N_parent = len(parent_labels)

    for subset in subsets:
        N_subset = len(subset)
        if N_subset > 0:
            weight = N_subset / N_parent
            weighted_subset_entropy += weight * calculate_entropy(subset)

    return parent_entropy - weighted_subset_entropy


# --- Tree Node Structure and Building ---

class Node:
    # Fixed constructor definition to satisfy Optional type hints
    def __init__(self, feature_index: Optional[int] = None, feature_name: Optional[str] = None,
                 value: Optional[int] = None, label: Optional[int] = None,
                 children: Optional[Dict[Any, 'Node']] = None):
        # For Decision Node
        self.feature_index = feature_index
        self.feature_name = feature_name
        self.children: Dict[Any, Node] = children if children is not None else {}
        # For Leaf Node
        self.value = value
        self.label = label


def build_decision_tree(data_indices: List[int], parent_labels: List[int], depth: int = 0, max_depth: int = 5) -> \
Optional[Node]:
    """Recursively builds the tree based on maximizing Information Gain."""

    if not data_indices:
        return None

    labels = [TARGETS[i] for i in data_indices]

    # 1. Termination Conditions
    if len(set(labels)) == 1:
        return Node(label=labels[0], value=labels[0])

    if depth >= max_depth:
        majority_label = Counter(labels).most_common(1)[0][0]
        return Node(label=majority_label, value=majority_label)

    # 2. Find Best Split
    best_gain = -1
    best_feature_index = -1
    best_feature_groups: Dict[Any, List[int]] = {}

    for feature_index in range(5):
        feature_groups = defaultdict(list)
        for data_idx in data_indices:
            feature_value = FEATURES[data_idx][feature_index]
            feature_groups[feature_value].append(data_idx)

        subsets_labels = [
            [TARGETS[idx] for idx in indices]
            for indices in feature_groups.values()
        ]

        gain = calculate_information_gain(labels, subsets_labels)

        if gain > best_gain:
            best_gain = gain
            best_feature_index = feature_index
            best_feature_groups = feature_groups

    # 3. Stop Condition C: If IG is too low (near zero)
    if best_gain <= 0.001:
        majority_label = Counter(labels).most_common(1)[0][0]
        return Node(label=majority_label, value=majority_label)

    # 4. Create Node and Recurse
    best_feature_name = FEATURE_NAMES[best_feature_index]
    current_node = Node(feature_index=best_feature_index, feature_name=best_feature_name)

    for value, indices in best_feature_groups.items():
        child_node = build_decision_tree(indices, labels, depth + 1, max_depth)
        if child_node:
            current_node.children[value] = child_node

    return current_node


def draw_tree(node: Node, prefix: str = "", is_last: bool = True):
    """Prints the tree structure textually."""
    if node is None:
        return

    marker = "└── " if is_last else "├── "

    if node.value is not None:
        print(f"{prefix}{marker}PREDICT: {node.value} Stars")
    else:
        print(f"{prefix}{marker}SPLIT on {node.feature_name} (Index {node.feature_index})")

        new_prefix = prefix + ("    " if is_last else "│   ")

        child_values = list(node.children.keys())
        for i, val in enumerate(child_values):
            child_node = node.children[val]
            is_last_child = (i == len(child_values) - 1)

            branch_marker = "├── " if not is_last_child else "└── "
            print(f"{new_prefix}{branch_marker}'{val}' --> ", end="")
            draw_tree(child_node, new_prefix + ("    " if is_last_child else "│   "), is_last_child)


def predict_tree(node: Node, sample: Tuple) -> int:
    """Predicts satisfaction using the trained Decision Tree."""
    if node.value is not None:
        return node.value

    feature_idx = node.feature_index
    feature_value = sample[feature_idx]

    if feature_value in node.children:
        return predict_tree(node.children[feature_value], sample)
    else:
        # Fallback
        return node.label if node.label is not None else 3

    # ===================================================================


# 4. MAIN EXECUTION AND COMPARISON
# ===================================================================

if __name__ == "__main__":

    print("--- Customer Satisfaction Predictor (Naive Bayes & Decision Tree) ---")

    # --- 1. Naive Bayes Implementation ---
    nb_model = NaiveBayesClassifier()
    nb_model.train(FEATURES, TARGETS)

    print("\n" + "=" * 30)
    print("NAIVE BAYES ANALYSIS")
    print("=" * 30)

    test_sample_1 = ("Fast", "Hot", 85, 4, 4)
    pred_nb_1 = nb_model.predict(test_sample_1)
    nb_model.explain_prediction(test_sample_1, pred_nb_1)

    test_sample_2 = ("Slow", "Cold", 75, 2, 1)
    pred_nb_2 = nb_model.predict(test_sample_2)
    nb_model.explain_prediction(test_sample_2, pred_nb_2)

    # --- 2. Decision Tree Implementation ---
    print("\n" + "=" * 30)
    print("DECISION TREE ANALYSIS")
    print("=" * 30)

    initial_indices = list(range(len(DUMMY_DATA)))
    initial_labels = TARGETS

    print("\n--- Tree Building Log (Max Depth 5) ---")
    dt_root = build_decision_tree(initial_indices, initial_labels, max_depth=5)

    print("\n--- Decision Tree Structure (Visualization) ---")
    if dt_root:
        draw_tree(dt_root)
    else:
        print("Tree failed to build.")

    print("\n--- Decision Tree Predictions ---")

    pred_dt_1 = predict_tree(dt_root, test_sample_1)
    print(f"Input {test_sample_1} -> Predicted Satisfaction: {pred_dt_1} Stars")

    pred_dt_2 = predict_tree(dt_root, test_sample_2)
    print(f"Input {test_sample_2} -> Predicted Satisfaction: {pred_dt_2} Stars")

    # --- 3. Comparison ---
    print("\n" + "=" * 30)
    print("MODEL COMPARISON")
    print("=" * 30)
    print(f"Test 1: NB Predicts {pred_nb_1} | DT Predicts {pred_dt_1}")
    print(f"Test 2: NB Predicts {pred_nb_2} | DT Predicts {pred_dt_2}")
    print("\nComparison Summary:")
    print("Naive Bayes uses probability density across all features simultaneously.")
    print("The Decision Tree creates sequential, hierarchical rules based on maximizing Information Gain at each step.")