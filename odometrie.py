import numpy as np
import matplotlib.pyplot as plt

# plots all given positions and their respective
# sigma-ellipsis given the covariance-matrices
def plot(positions, kovP):

    plt.figure(figsize=(10, 10), dpi=120)

    i = 0

    # t = {0, 2 * pi}
    t = np.linspace(0, 2 * np.pi, 100)

    # iterate over all positions
    for pos in positions:

        # plot the position
        plt.plot(pos[0], pos[1], marker='o', markersize=2, color="red")

        # skip first iteration, because kovP[0] is a singular matrix
        if i > 0:
            # calculate eigenvalues and eigenvectors
            eigenValues, eigenVectors = np.linalg.eig(np.linalg.inv(kovP[i][0:2, 0:2]))

            # calculate the set of points for the ellipsis
            # for t = {0, 2 * pi}
            ellipsis = np.array((
                (1/np.sqrt(eigenValues[0]) * np.cos(t)),
                (1/np.sqrt(eigenValues[1]) * np.sin(t))
            ))

            # rotate and move ellipsis
            # with V = [ v1, v2 ]
            # and Mu = [X, Y]
            ellipsis_x, ellipsis_y = np.transpose(
                np.add(
                    np.transpose(
                        np.matmul(eigenVectors, ellipsis)
                    ),
                    pos[0:2]
                )
            )

            # plot ellipsis
            plt.plot(ellipsis_x, ellipsis_y, '-', color='blue', linewidth=1)

        i = i + 1

    # graph-properties
    plt.xticks(np.arange(-1.5, 3.5, 0.1))
    plt.yticks(np.arange(-1.5, 3.5, 0.1))

    plt.grid(color='#cccccc', linestyle='-', linewidth=0.5)

    plt.title('Lokalisierung mittels Odometrie - Fehlerellipsen\n', fontsize=16)

    plt.xlabel("$P_n[X]$")
    plt.ylabel("$P_n[Y]$")

    plt.show()
    return

# calculates the positions and covariance-matrices
# given a start position with (x, y, theta),
# the distance of the wheels (d), the factor of uncertainty (k)
# and a path {(distance, rotation), ...}
def calc(x, y, theta, d, k, path):

    # position-array
    pos = np.array(
        [
            [x, y, theta]
        ]
    )

    # covariance-matrix P
    kovP = np.array(
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
        ]
    )

    i = 0

    # iterate over all steps in our defined path
    for step in path:

        # distance driven
        delta_s = step[0]

        # change of direction
        delta_theta = step[1]

        # x position at point in time t
        x_n = pos[i][0]

        # y position at point in time t
        y_n = pos[i][1]

        # rotation at point in time t
        theta_n = pos[i][2]

        # calculate the new position
        resultPos = np.array(
            [
                np.add(
                    np.array([
                        x_n,
                        y_n,
                        theta_n
                    ]),
                    np.array([
                        delta_s * np.cos(theta_n + (delta_theta/2)),
                        delta_s * np.sin(theta_n + (delta_theta/2)),
                        delta_theta
                    ])
                )
            ]
        )

        # append the new position to our position-array
        pos = np.append(pos, resultPos, axis=0)

        # calculate jacobi-matrix Gp
        GP = np.array(
            [
                [
                    1, 0, -delta_s * np.sin(theta_n + (delta_theta/2))
                ],
                [
                    0, 1, delta_s * np.cos(theta_n + (delta_theta/2))
                ],
                [
                    0, 0, 1
                ]
            ]
        )

        # calculate jacobi-matrix Gs
        GS = np.array(
            [
                [
                    (0.5 * np.cos(theta_n + (delta_theta/2))) - ((delta_s / (2 * d)) * np.sin(theta_n + (delta_theta/2))),
                    (0.5 * np.cos(theta_n + (delta_theta/2))) + ((delta_s / (2 * d)) * np.sin(theta_n + (delta_theta/2)))
                ],
                [
                    (0.5 * np.sin(theta_n + (delta_theta/2))) + ((delta_s / (2 * d)) * np.cos(theta_n + (delta_theta/2))),
                    (0.5 * np.sin(theta_n + (delta_theta/2))) - ((delta_s / (2 * d)) * np.cos(theta_n + (delta_theta/2)))
                ],
                [
                    1 / d,
                    -1 / d
                ]
            ]
        )

        # calculate covariance-matrix S
        kovS = np.array(
            [
                [k * np.absolute(delta_s), 0],
                [0, k * np.absolute(delta_s)]
            ]
        )

        # calculate final covariance-matrix
        resultKov = np.array(
            [
                np.add(
                    np.matmul(
                        np.matmul(GP, kovP[i]),
                        np.transpose(GP)
                    ),
                    np.matmul(
                        np.matmul(GS, kovS),
                        np.transpose(GS)
                    )
                )
            ]
        )

        # append the result to our array of covariance-matrices
        kovP = np.append(kovP, resultKov, 0)

        i = i + 1
    return pos, kovP


# factor of uncertainty
k = 0.001

# spacing between wheels in meter
d = 0.20

# start position
x = 0
y = 0
theta = 0

# path:
# 5 steps, each 20 cm forward and a rotation of pi/10,
# followed by 5 steps with each 15cm forward and a rotation of pi/10
path = np.array([
    (0.2, np.pi / 10),
    (0.2, np.pi / 10),
    (0.2, np.pi / 10),
    (0.2, np.pi / 10),
    (0.2, np.pi / 10),
    (0.15, np.pi / 10),
    (0.15, np.pi / 10),
    (0.15, np.pi / 10),
    (0.15, np.pi / 10),
    (0.15, np.pi / 10)
])

# calculate the positions and covariances
positions, covariances = calc(x, y, theta, d, k, path)

# plot the result
plot(positions, covariances)
