import numpy as np

obj = {}


def RSR(X, lambda_):
    iter_num = 50
    (n, m) = np.shape(X)

    # initialize Gr and Gl
    Gr = np.ones((m, 1))
    Gl = np.ones((n, 1))

    if m > n:
        for iter in range(iter_num):
            G_R = np.diag(Gr)
            G_L = np.diag(Gl)

            # update W
            W = np.linalg.inv(
                (G_R * X.T * G_L * X + lambda_ * np.eye(m))) * (G_R * X.T * G_L * X)

            # update Gr
            wc = np.sqrt(np.sum(W * W, axis=1))
            Gr = 2 * wc

            # update Gl
            E = X * W - X
            ec = np.sqrt(np.sum(E * E, axis=1) + 0.00001)  # np.eps
            Gl = 0.5 / ec

            obj[iter] = sum(ec) + lambda_ * sum(wc)
    else:
        for iter in range(iter_num):
            G_R = np.diag(Gr)
            G_L = np.diag(Gl)

            # update W
            W = G_R * X.T * G_L * \
                ((X * G_R * X.T * G_L + lambda_ * np.eye(n)) / X)

            # update Gr
            wc = np.sqrt(np.sum(W * W, axis=1))
            Gr = 2 * wc

            # update Gl
            E = X * W - X
            ec = np.sqrt(np.sum(E * E, axis=1) + 0.00001)  # np.eps
            Gl = 0.5 / ec

            obj[iter] = sum(ec) + lambda_ * sum(wc)
    return obj


# Example of how to use the function
a = range(15)
b = range(15, 30)
c = range(30, 45)
my_data = [list(a), list(b), list(c)]

my_data = np.array(my_data)

RSR(my_data, 0.1)

