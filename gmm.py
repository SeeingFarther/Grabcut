from sklearn.cluster import KMeans
import numpy as np
import cv2
np.seterr(divide='ignore', invalid='ignore')

# GMM utility for grabcut algorithm based on the papers:
# Paper 1 - Carsten Rother, Vladimir Kolmogorov, and Andrew Blake. “” GrabCut” interactive
# foreground extraction using iterated graph cuts”. In: ACM transactions on graphics (TOG) 23.3 (2004), pp. 309–314
# Paper 2 - Justin F Talbot and Xiaoqian Xu. “Implementing grabcut”. In: Brigham
# Young University 3 (2006).


class GMM:

    # Init the GMM
    def __init__(self, points, n_components):
        self.n_components = n_components
        self.components_index = []
        self.means = np.zeros((self.n_components, 3), dtype=np.float64)
        self.covariances = np.zeros((self.n_components, 3, 3), dtype=np.float64)
        self.invert_covariances = np.zeros((self.n_components, 3, 3), dtype=np.float64)
        self.det = np.zeros(self.n_components, dtype=np.float64)
        self.phi = np.zeros(self.n_components, dtype=np.float64)

        # Init fit using k-means algorithm
        kmeans = KMeans(n_clusters=self.n_components, init='k-means++', n_init=1, random_state=42).fit(points)
        self.components_index = kmeans.labels_

        # Update GMM parameters
        self.update_params(points, self.components_index)
        return

    # Update the GMM mean and covariance and other parameters
    def update_params(self, points, components_index):

        for i in range(self.n_components):
            data = points[components_index == i]

            # no points in the GMM?
            if len(data) == 0:
                self.phi[i] = 0
                self.covariances[i] = np.zeros((3, 3), dtype=np.float64)
                self.means[i] = np.zeros(3, dtype=np.float64)
                self.invert_covariances[i] = np.zeros((3, 3), dtype=np.float64)
                self.det[i] = 1
                continue

            # GMM parameters
            cov, mean = cv2.calcCovarMatrix(samples=data,flags=cv2.COVAR_SCALE | cv2.COVAR_NORMAL | cv2.COVAR_ROWS, mean=None)
            self.means[i] = mean
            self.covariances[i] = cov

            self.det[i] = np.linalg.det(self.covariances[i])
            self.phi[i] = data.shape[0] / points.shape[0]

            # Singular matrix? solve
            if self.det[i] <= 0:

                # Create non-singular matrix which will be added to covariance matrix
                # We create covariance matrix with small values as stated in paper 2
                dim = 3
                m = np.identity(dim) * 0.25  # Small numbers inside the matrix
                mx = np.sum(np.abs(m), axis=1)
                np.fill_diagonal(m, mx)

                self.covariances[i] += m
                self.det[i] = np.linalg.det(self.covariances[i])

            if self.det[i] > 0:
                self.invert_covariances[i] = np.linalg.inv(self.covariances[i])

    # Calculate the probabilities of the points for specific Gaussian component of the GMM without normalizing
    def compute_prob(self, points, component):
        probabilities = np.zeros(points.shape[0], dtype=np.float64)

        # No points in the gaussian component or singular matrix? send zero probabilities vector for all points
        if self.phi[component] == 0 or self.det[component] <= 0:
            return probabilities

        # Matrix multiplication for the exponent in gaussian probability,
        # Einstein notation is faster and prevent overflow for big matrices which we had with np.matmul
        normalized_points = points - self.means[component]
        z = np.einsum("ij, jk -> ik", normalized_points, self.invert_covariances[component])
        alpha = np.einsum('ij, ij -> i', normalized_points, z)

        # Calculate the probabilities using the gaussian probability formula
        exponent = alpha / 2
        denominator = 1 / np.sqrt(np.power(2 * np.pi, 3) * self.det[component])
        probabilities = np.exp(-exponent) * denominator
        return probabilities

    # Given Samples predict their labels
    def predict(self, points):
        # No points? send empty array
        if points.size == 0:
            return np.empty(shape=(0, 0))

        # For each point calculate the probability to be part of the "i" component
        num_of_points = len(points)
        prob = np.empty(shape=(len(points), 0))
        for i in range(self.n_components):
            y = self.compute_prob(points, i).reshape(num_of_points, 1)
            prob = np.concatenate((prob, y), axis=1)

        # Choose for each point the component with the max probability
        return np.argmax(prob, axis=1)

    # Update the GMM parameter according to the points entered
    def update(self, points):
        # No points? send empty array
        if points.size == 0:
            return np.empty(shape=(0, 0))

        self.components_index = self.predict(points)
        self.update_params(points, self.components_index)

    # Compute the inner likelihood(before running the log),
    # which will be summed together in the D function like in Paper 2
    def inner_likelihood(self, points, component):
        # No points in the gaussian component or singular matrix? send zero inner likelihood vector for all points
        inner_likelihood = np.zeros(points.shape[0], dtype=np.float64)
        if self.phi[component] == 0 or self.det[component] <= 0:
            return inner_likelihood

        # Matrix multiplication for the exponent in gaussian probability,
        # Einstein notation is faster and prevent overflow for big matrices which we had with np.matmul
        normalized_points = points - self.means[component]
        z = np.einsum("ij, jk -> ik", normalized_points, self.invert_covariances[component])
        alpha = np.einsum('ij, ij -> i', normalized_points, z)

        # Calculate the inner likelihood(before running the log)
        exponent = alpha / 2
        denominator = (1 / (np.sqrt(self.det[component])))
        inner_likelihood = np.exp(-exponent) * denominator

        return inner_likelihood

    # Compute the "d" function value like specify in the paper 2
    def compute_d_function(self, points):
        if points.size == 0:
            return np.empty(shape=(0, 0))

        # For each point calculate the inner_likelihood of the "i" component
        num_of_points = len(points)
        prob = np.empty(shape=(len(points), 0))
        for i in range(self.n_components):
            y = self.inner_likelihood(points, i).reshape(num_of_points, 1)
            prob = np.concatenate((prob, y), axis=1)

        # Matrix multiplication with vector
        delta = np.einsum('ij, j -> i', prob, self.phi)

        # Return the "d" function like specify in the paper 2
        return -np.log(delta)



