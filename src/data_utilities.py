import numpy as np


class DataParameters:
    def __init__(self):
        self.num_points = 4
        self.poly_degree = 2
        self.interval_min = -1.0
        self.interval_max = 1.0
        self.is_random = False

    def __str__(self):
        string = 'DataParameters: \n'
        string += '    num_points: {}\n'.format(self.num_points)
        string += '    poly_degree: {}\n'.format(self.poly_degree)
        string += '    interval_min: {}\n'.format(self.interval_min)
        string += '    interval_max: {}\n'.format(self.interval_max)
        string += '    is_random: {}\n'.format(self.is_random)
        return string


class DataPolynomial:

    def __init__(self, params: DataParameters):
        # The convention is x for the value on x-axis. phi is the features phi(x)

        if params.is_random is False:
            self.x = np.linspace(params.interval_min, params.interval_max, params.num_points)
            self.x[::2] += 0.1

            ones = np.ones(self.x.shape)
            ones[::2] = -1

            # create y as column vector
            self.y = (self.x + ones)[:, np.newaxis]
            self.y[::2] += 0.1
        else:
            self.x = np.random.uniform(low=params.interval_min, high=params.interval_min, size=params.num_points)
            self.x -= self.x.mean()

            # y is column vector
            self.y = np.random.uniform(low=-1, high=1, size=params.num_points)[:, np.newaxis]

        # variables that related to polynomial degree
        self.phi_train = None
        self.theta_erm = None
        self.poly_degree = None

    def create_train(self, poly_degree: int):
        self.poly_degree = poly_degree

        # Create Feature matrix
        self.phi_train = self.convert_to_features(poly_degree)

        # ERM estimator
        self.theta_erm = self.fit_least_squares_estimator(self.phi_train, self.get_labels_array())

    def convert_to_features(self, pol_degree):
        # phy_i = [x_i^0 , x_i^1, ... , x_1^pol_degree].T
        # phy = [phy_1, phy_2, phy_3 ... ]
        phi = []
        for n in range(pol_degree + 1):
            phi.append(np.power(self.x, n))
        phi = np.asarray(phi)
        return phi

    @staticmethod
    def convert_point_to_features(x, pol_degree):
        # phi = [x^0, x^1, ... x^pol_degree].T
        phi = []
        for n in range(pol_degree + 1):
            phi.append(np.power(x, n))
        phi = np.asarray(phi)

        if len(phi.shape) == 1:
            phi = phi[:, np.newaxis]

        return phi

    def get_data_points_as_list(self):
        return self.x.tolist(), self.y.tolist()

    def get_labels_array(self):
        return self.y

    @staticmethod
    def fit_least_squares_estimator(phi: np.ndarray, y: np.ndarray, lamb: float = 0.0):
        phi_phi_t = np.matmul(phi, phi.transpose())
        phi_phi_t_inv = np.linalg.pinv(phi_phi_t + lamb * np.eye(phi_phi_t.shape[0], phi_phi_t.shape[1]))
        theta = phi_phi_t_inv.dot(phi).dot(y)
        return theta

    @staticmethod
    def predict_from_least_squares(theta: np.ndarray, phy_test: np.ndarray):
        y_predicted = phy_test.T.dot(theta)
        assert y_predicted.shape[1] == 1
        return y_predicted

    @staticmethod
    def add_test_to_train(phi_train, phi_test):
        return np.concatenate((phi_train, phi_test), axis=1)
