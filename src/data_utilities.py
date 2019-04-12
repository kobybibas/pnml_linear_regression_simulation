import numpy as np


class DataParameters:
    """
    Class of data related parameters.
    """

    def __init__(self):
        # Number of training data.
        self.num_points = 4

        # Default polynomial degree to fit.
        self.poly_degree = 2

        # In case of random training points: the interval on which to generate.
        self.is_random = False
        self.interval_min = -1.0
        self.interval_max = 1.0

        # Define specific training point.
        self.x_train = None
        self.y_train = None

    def __str__(self):
        string = 'DataParameters: \n'
        string += '    num_points: {}\n'.format(self.num_points)
        string += '    poly_degree: {}\n'.format(self.poly_degree)
        string += '    is_random: {}\n'.format(self.is_random)
        string += '    interval_min: {}\n'.format(self.interval_min)
        string += '    interval_max: {}\n'.format(self.interval_max)
        string += '    x_train: {}\n'.format(self.x_train)
        string += '    y_train: {}\n'.format(self.y_train)
        return string


class DataPolynomial:

    def __init__(self, params: DataParameters):
        """
        Create the training data.
        :param params: parameters for creating the training data.
        """

        # The polynomial degree that will be fitted to the data.
        self.poly_degree = None

        # Matrix of training feature [phi0;phi1;phi2...]. phi is the features phi(x)
        self.phi_train = None

        # The least squares empirical risk minimization solution
        self.theta_erm = None

        # Generate the training data.
        if params.x_train is not None and params.y_train is not None:
            self.x = np.array(params.x_train)
            self.y = np.array(params.y_train)
            return

        if params.is_random is False:
            self.x = np.linspace(params.interval_min, params.interval_max, params.num_points)
            self.x[::2] += 0.1

            ones = np.ones(self.x.shape)
            ones[::2] = -1

            # create y as column vector
            self.y = (self.x + ones)[:, np.newaxis]
            self.y[::2] += 0.1
        else:
            self.x = np.random.uniform(low=params.interval_min, high=params.interval_max, size=params.num_points)
            # self.x -= self.x.mean()

            # y is column vector
            self.y = np.random.uniform(low=-1, high=1, size=params.num_points)[:, np.newaxis]

    def create_train(self, poly_degree: int):
        """
        Convert data points to feature matrix: phi=[x0^0,x0^1,x0^2...;x1^0,x1^1,x1^2...;x2^0,x2^1,x2^2...]
        :param poly_degree: the assumed polynomial degree of the training set.
        :return: phi: training set feature matrix.
        """
        self.poly_degree = poly_degree

        # Create Feature matrix
        self.phi_train = self.convert_to_features(poly_degree)

    def convert_to_features(self, pol_degree: int) -> np.ndarray:
        """
        Convert the training point to feature matrix.
        :param pol_degree: the assumed polynomial degree of the data.
        :return: phy = [x0^0 , x0^1, ... , x0^pol_degree; x1^0 , x1^1, ... , x1^pol_degree].T
        """
        phi = []
        for n in range(pol_degree + 1):
            phi.append(np.power(self.x, n))
        phi = np.asarray(phi)
        return phi

    @staticmethod
    def convert_point_to_features(x: np.ndarray, pol_degree: int) -> np.ndarray:
        """
        Given a training point, convert it to features
        :param x: training point.
        :param pol_degree: the assumed polynomial degree.
        :return: phi = [x^0,x^1,x^2,...].T
        """
        phi = []
        for n in range(pol_degree + 1):
            phi.append(np.power(x, n))
        phi = np.asarray(phi)

        if len(phi.shape) == 1:
            phi = phi[:, np.newaxis]

        return phi

    def get_data_points_as_list(self):
        """
        :return: list of training set data. list of training set labels.
        """
        return self.x.tolist(), self.y.tolist()

    def get_labels_array(self) -> np.ndarray:
        """
        :return: the labels of the training set.
        """
        return self.y

    @staticmethod
    def fit_least_squares_estimator(phi: np.ndarray, y: np.ndarray, lamb: float = 0.0) -> np.ndarray:
        """
        Fit ERM least squares estimator
        :param phi: the training set features matrix.
        :param y: the labels vector.
        :param lamb: regularization term.
        :return: the fitted parameters.
        """
        phi_phi_t = np.matmul(phi, phi.transpose())
        phi_phi_t_inv = np.linalg.pinv(phi_phi_t + lamb * np.eye(phi_phi_t.shape[0], phi_phi_t.shape[1]))
        theta = phi_phi_t_inv.dot(phi).dot(y)
        return theta

    @staticmethod
    def predict_from_least_squares(theta: np.ndarray, phy_test: np.ndarray) -> np.ndarray:
        """
        Predict the label of the test point.
        :param theta: fitted least squared parameters.
        :param phy_test: feature of test point on which to predict.
        :return: the prediction of the test point.
        """
        y_predicted = phy_test.T.dot(theta)
        assert y_predicted.shape[1] == 1
        return y_predicted

    @staticmethod
    def add_test_to_train(phi_train: np.ndarray, phi_test: np.ndarray) -> np.ndarray:
        """
        Add the test set feature to training set feature matrix
        :param phi_train: training set feature matrix.
        :param phi_test: test set feature.
        :return: concat train and test.
        """
        return np.concatenate((phi_train, phi_test), axis=1)
