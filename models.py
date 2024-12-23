import numpy as np

class LDAModel:
    def fit(self, X, y):
        """
        Fits the model to a given training dataset.
        This method learns the parameters of the model from the provided data, enabling it to make predictions on unseen examples.
        Args:
        X (ndarray): The training features, a 2D array of shape (n_samples, n_features).
        y (ndarray): The training labels, a 1D array of shape (n_samples,)."""
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.unique_classes = unique_classes
        self.class_counts = class_counts
        num_classes = len(unique_classes)
        self.num_classes = num_classes
        self.class_means = np.array([np.mean(X[y == c], axis=0) for c in unique_classes])
        covariance_matrices = [np.cov(X[y == c], rowvar=False) for c in unique_classes]
        weights = class_counts.astype(float) / len(y)
        self.shared_covariance_matrix = np.sum([weights[i] * covariance_matrices[i] for i in range(num_classes)], axis=0)

    def predict(self, X):
        """
        Predicts labels or values for new data samples.
        This method uses the model trained in `fit` to generate predictions for unseen data.
        Args:
            X (ndarray): The input features, a 2D array of shape (n_samples, n_features).
        Returns:
            ndarray: The predicted labels or values, a 1D array of shape (n_samples,)."""
        y_pred = []
        num_classes = self.num_classes
        inv_covariance_matrix = np.linalg.inv(self.shared_covariance_matrix)
        constant_terms = [-0.5 * self.class_means[i].T @ inv_covariance_matrix @ self.class_means[i] for i in range(num_classes)]
        for x in X:
            discriminant_values = [x.T @ inv_covariance_matrix @ self.class_means[i] + constant_terms[i] for i in range(num_classes)]
            predicted_class = np.argmax(discriminant_values)
            y_pred.append(predicted_class)
        return y_pred

class QDAModel:
    def fit(self, X, y):
        """
        Fits the model to a given training dataset.
        This method learns the parameters of the model from the provided data, enabling it to make predictions on unseen examples.
        Args:
        X (ndarray): The training features, a 2D array of shape (n_samples, n_features).
        y (ndarray): The training labels, a 1D array of shape (n_samples,)."""
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.unique_classes = unique_classes
        self.class_counts = class_counts
        self.num_classes = len(unique_classes)
        total_samples = len(y)
        self.log_priors = np.log(np.array(class_counts) / total_samples)
        self.class_means = np.array([np.mean(X[y == c], axis=0) for c in unique_classes])
        self.class_covariance_matrices = [np.cov(X[y == c], rowvar=False) for c in unique_classes]

    def predict(self, X):
        """
        Predicts labels or values for new data samples.
        This method uses the model trained in `fit` to generate predictions for unseen data.
        Args:
            X (ndarray): The input features, a 2D array of shape (n_samples, n_features).
        Returns:
            ndarray: The predicted labels or values, a 1D array of shape (n_samples,)."""
        y_pred = []
        num_classes = self.num_classes
        inv_covariance_matrices = [np.linalg.inv(self.class_covariance_matrices[i]) for i in range(num_classes)]
        constant_terms = [(- 0.5 * np.exp(np.log(np.linalg.slogdet(self.class_covariance_matrices[i])[1])) + self.log_priors[i]) for i in range(num_classes)]
        for x in X:
            discriminant_values = []
            for i in range(num_classes):
                diff = x - self.class_means[i]
                discriminant_value = -0.5 * (diff.T @ inv_covariance_matrices[i] @ diff) + constant_terms[i]
                discriminant_values.append(discriminant_value)
            predicted_class = np.argmax(discriminant_values)
            y_pred.append(predicted_class)
        return np.array(y_pred)


class GaussianNBModel:
    def fit(self, X, y):
        """
        Fits the model to a given training dataset.
        This method learns the parameters of the model from the provided data, enabling it to make predictions on unseen examples.
        Args:
        X (ndarray): The training features, a 2D array of shape (n_samples, n_features).
        y (ndarray): The training labels, a 1D array of shape (n_samples,)."""
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.unique_classes = unique_classes
        self.class_counts = class_counts
        self.num_classes = len(unique_classes)
        total_samples = X.shape[0]
        self.log_priors = np.log(np.array(class_counts) / total_samples)
        self.class_means = np.array([np.mean(X[y == c], axis=0) for c in range(self.num_classes)])
        self.class_vars = np.array([np.var(X[y==c], axis=0) for c in range(self.num_classes)])

    def predict(self, X):
        """
        Predicts labels or values for new data samples.
        This method uses the model trained in `fit` to generate predictions for unseen data.
        Args:
            X (ndarray): The input features, a 2D array of shape (n_samples, n_features).
        Returns:
            ndarray: The predicted labels or values, a 1D array of shape (n_samples,)."""
        y_pred = []
        for x in X:
            predicted_class = np.argmax([self.log_priors[i] + np.sum(-0.5*np.log(2*np.pi*self.class_vars[i]) - 0.5*((x-self.class_means[i])**2)/self.class_vars[i]) for i in range(self.num_classes)])
            y_pred.append(predicted_class)
        return y_pred
    