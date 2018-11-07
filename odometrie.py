from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math


def plot(positions, kovP):

    plt.figure(figsize=(10, 10), dpi=120)

    i = 0

    t = np.linspace(0, 2 * math.pi, 100)

    for pos in positions:

        plt.plot(pos[0], pos[1], marker='o', markersize=2, color="red")

        if i > 0:
            eigenValues, eigenVectors = np.linalg.eig(np.linalg.inv(kovP[i][0:2, 0:2]))

            ellipsis = np.array((
                (1/np.sqrt(eigenValues[0]) * np.cos(t)),
                (1/np.sqrt(eigenValues[1]) * np.sin(t))
            ))

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

    plt.xticks(np.arange(-0.1, 1.5, 0.1))
    plt.yticks(np.arange(-0.1, 1.5, 0.1))

    plt.grid(color='#cccccc', linestyle='-', linewidth=0.5)

    plt.title('Lokalisierung mittels Odometrie - Fehlerellipsen\n', fontsize=16)

    plt.xlabel("$P_n[X]$")
    plt.ylabel("$P_n[Y]$")

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
                    1, 0, -delta_s * math.sin(theta_n)
                ],
                [
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
                    (0.5 * math.cos(theta_n)) - ((delta_s / (2 * d)) * math.sin(theta_n)),
                    (0.5 * math.cos(theta_n)) + ((delta_s / (2 * d)) * math.sin(theta_n))
                ],
                [
                    (0.5 * math.sin(theta_n)) + ((delta_s / (2 * d)) * math.cos(theta_n)),
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
