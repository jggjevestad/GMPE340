from numpy import append, zeros
from scipy.linalg import expm


def numeval(F, G, dt):

    # Matrix size
    [m, n] = F.shape

    # Build matrix (Van Loan)
    A1 = append(-F, G@G.T, axis=1)
    A2 = append(zeros([m, n]), F.T, axis=1)
    A = append(A1, A2, axis=0)*dt

    # Compute matrix exponential
    B = expm(A)

    # Extract phi and Q
    phi = B[m:m*2, n:n*2].T
    Q = phi@B[0:m, n:n*2]

    return phi, Q


# Example
if __name__ == '__main__':

    # Import libraries
    from numpy import array

    # Time interval
    dt_ = 0.1

    # Dynamic matrix
    F_ = array([[0, 1],
                [0, 0]])
    print(F_)

    # White noise coefficients
    G_ = array([[0],
                [1.25]])
    print(G_)

    # Numerical evaluation
    phi_, Q_ = numeval(F_, G_, dt_)
    print(phi_)
    print(Q_)
