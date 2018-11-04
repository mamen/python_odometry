from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import math
from matplotlib.patches import Ellipse


def get_cov_ellipse(ax, cov, centre):
    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(cov)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)

    ellipse = Ellipse(centre, w, h, theta, color="blue", fill=False)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(1)
    ax.add_artist(ellipse)


def plot(positions, kovP):

    minX, maxX, minY, maxY = 0, 0, 0, 0

    for pos in positions:

        if pos[0] < minX:
            minX = pos[0]

        if pos[0] > maxX:
            maxX = pos[0]

        if pos[1] < minY:
            minY = pos[1]

        if pos[1] > maxY:
            maxY = pos[1]

    plt.figure(figsize=(10, 10), dpi=120)

    # Create a new subplot from a grid of 1x1
    plt.subplot(1, 1, 1)

    i = 0

    ax = plt.gca()

    for pos in positions:
        plt.plot(pos[0], pos[1], marker='o', markersize=2, color="red")

        if i > 0:
            eigenValues, eigenVectors = np.linalg.eig(np.linalg.inv(kovP[i][0:2, 0:2]))

            print eigenVectors

            # print eigenValues[0], eigenValues[1]
            # print eigenVectors
            # print "----"

            # ellipse = Ellipse(
            #     xy=(pos[0], pos[1]),
            #     width=2 * math.sqrt(eigenValues[0]),
            #     height=2 * math.sqrt(eigenValues[1]),
            #     angle=np.degrees(np.arctan(eigenVectors[0], eigenVectors[1])[0]),
            #     linewidth=0.5,
            #     fill=False,
            #     zorder=2
            # )
            #
            # ax.add_patch(ellipse)

            #def get_cov_ellipse(cov, centre, nstd, **kwargs):

        get_cov_ellipse(ax, kovP[i][0:2,0:2], (pos[0], pos[1]))

        i = i + 1

    # Set x limits
    plt.xlim(-0.1, 1.3)

    # Set y limits
    plt.ylim(-0.1, 1.3)

    # Set x ticks
    # plt.xticks(np.linspace(1, 100, endpoint=True))

    totalMin = min(minX, minY)
    totalMax = max(maxX, maxY)

    plt.xticks(np.arange(-0.1, 1.3, 0.1))
    plt.yticks(np.arange(-0.1, 1.3, 0.1))

    plt.grid(color='#cccccc', linestyle='-', linewidth=0.5)

    # Set y ticks
    # plt.yticks(np.linspace(-1, 1, 5, endpoint=True))

    # Save figure using 72 dots per inch
    # plt.savefig("exercice_2.png", dpi=72)

    # Show result on screen
    plt.show()
    return


def calc(x, y, theta, d, k, path):
    pos = np.array(
        [
            [x, y, theta]
        ]
    )

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

    for step in path:

        # gefahrene wegstrecke
        delta_s = step[0]

        # richtungsaenderung
        delta_theta = step[1]

        x_n = pos[i][0]
        y_n = pos[i][1]
        theta_n = pos[i][2]

        resultPos = np.array(
            [
                np.add(
                    np.array([
                        x_n,
                        y_n,
                        theta_n
                    ]),
                    np.array([
                        delta_s * math.cos(theta_n),
                        delta_s * math.sin(theta_n),
                        delta_theta
                    ])
                )
            ]
        )

        pos = np.append(pos, resultPos, axis=0)

        GP = np.array(
            [
                [
                    #1, 0, -delta_s * math.sin(theta_n + (delta_theta / 2))
                    1, 0, -delta_s * math.sin(theta_n)
                ],
                [
                    #0, 1, delta_s * math.cos(theta_n + (delta_theta / 2))
                    0, 1, delta_s * math.cos(theta_n)
                ],
                [
                    0, 0, 1
                ]
            ]
        )

        GS = np.array(
            [
                [
                    #(0.5 * math.cos(theta_n + (delta_theta / 2))) - ((delta_s / (2 * d)) * math.sin(theta_n + (delta_theta / 2))),
                    (0.5 * math.cos(theta_n)) - ((delta_s / (2 * d)) * math.sin(theta_n)),
                    #(0.5 * math.cos(theta_n + (delta_theta / 2))) + ((delta_s / (2 * d)) * math.sin(theta_n + (delta_theta / 2)))
                    (0.5 * math.cos(theta_n)) + ((delta_s / (2 * d)) * math.sin(theta_n))
                ],
                [
                    #(0.5 * math.sin(pos[i - 1][2] + (step[1] / 2))) + ((step[0] / (2 * d)) * math.cos(pos[i - 1][2] + (step[1] / 2))),
                    (0.5 * math.sin(theta_n)) + ((delta_s / (2 * d)) * math.cos(theta_n)),
                    #(0.5 * math.sin(pos[i - 1][2] + (step[1] / 2))) - ((step[0] / (2 * d)) * math.cos(pos[i - 1][2] + (step[1] / 2)))
                    (0.5 * math.sin(theta_n)) - ((delta_s / (2 * d)) * math.cos(theta_n))
                ],
                [
                    1 / d,
                    -1 / d
                ]
            ]
        )

        kovS = np.array(
            [
                [k * np.absolute(delta_s), 0],
                [0, k * np.absolute(delta_s)]
            ]
        )

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

        kovP = np.append(kovP, resultKov, 0)

        i = i + 1
    return pos, kovP


# Unsicherheitsfaktor
k = 0.001

# Radabstand
d = 0.20

# Startposition
x = 0
y = 0
theta = 0

# Pfad: 5 Schritte mit jeweils 20 cm vorwaerts mit je einer Drehung von pi/10, dann 5 Schritte mit jeweils
# 15 cm vorwaerts mit je einer Drehung von pi/10
path = np.array([
    (0.2, math.pi / 10),
    (0.2, math.pi / 10),
    (0.2, math.pi / 10),
    (0.2, math.pi / 10),
    (0.2, math.pi / 10),
    (0.15, math.pi / 10),
    (0.15, math.pi / 10),
    (0.15, math.pi / 10),
    (0.15, math.pi / 10),
    (0.15, math.pi / 10)
])

positions, kovariances = calc(x, y, theta, d, k, path)

plot(positions, kovariances)
