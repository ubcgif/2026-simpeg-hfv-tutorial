import numpy as np


def computeH0(tx_nodes, xyz):
    """
    Computes inducing field at locations xyz

    REQUIRED ARGUMENTS:

    xyz -- N X 3 array of locations at which primary field components
    are computed

    OUTPUTS:

    H0 -- N X 3 array containing [Hx0,Hy0,Hz0] at all xyz locations

    """

    # TRANSMITTER NODES
    x1tr = tx_nodes[:, 0]
    x2tr = tx_nodes[:, 1]
    x3tr = tx_nodes[:, 2]

    nLoc = np.shape(xyz)[0]
    nSeg = np.size(x1tr) - 1

    hx0 = np.zeros(nLoc)
    hy0 = np.zeros(nLoc)
    hz0 = np.zeros(nLoc)

    for pp in range(0, nSeg):

        # Wire ends for transmitter wire pp
        x1a = x1tr[pp]
        x2a = x2tr[pp]
        x3a = x3tr[pp]
        x1b = x1tr[pp + 1]
        x2b = x2tr[pp + 1]
        x3b = x3tr[pp + 1]

        # Vector Lengths between points
        vab = np.sqrt((x1b - x1a) ** 2 + (x2b - x2a) ** 2 + (x3b - x3a) ** 2)
        vap = np.sqrt(
            (xyz[:, 0] - x1a) ** 2 + (xyz[:, 1] - x2a) ** 2 + (xyz[:, 2] - x3a) ** 2
        )
        vbp = np.sqrt(
            (xyz[:, 0] - x1b) ** 2 + (xyz[:, 1] - x2b) ** 2 + (xyz[:, 2] - x3b) ** 2
        )

        # Cosines from cos()=<v1,v2>/(|v1||v2|)
        cos_alpha = (
            (xyz[:, 0] - x1a) * (x1b - x1a)
            + (xyz[:, 1] - x2a) * (x2b - x2a)
            + (xyz[:, 2] - x3a) * (x3b - x3a)
        ) / (vap * vab)
        cos_beta = (
            (xyz[:, 0] - x1b) * (x1a - x1b)
            + (xyz[:, 1] - x2b) * (x2a - x2b)
            + (xyz[:, 2] - x3b) * (x3a - x3b)
        ) / (vbp * vab)

        # Determining Radial Vector From Wire
        dot_temp = (
            (x1a - xyz[:, 0]) * (x1b - x1a)
            + (x2a - xyz[:, 1]) * (x2b - x2a)
            + (x3a - xyz[:, 2]) * (x3b - x3a)
        )

        rx1 = (x1a - xyz[:, 0]) - dot_temp * (x1b - x1a) / vab**2
        rx2 = (x2a - xyz[:, 1]) - dot_temp * (x2b - x2a) / vab**2
        rx3 = (x3a - xyz[:, 2]) - dot_temp * (x3b - x3a) / vab**2

        r = np.sqrt(rx1**2 + rx2**2 + rx3**2)

        phi = (cos_alpha + cos_beta) / r

        # I/4*pi in each direction
        ix1 = (x1b - x1a) / (4 * np.pi * vab)
        ix2 = (x2b - x2a) / (4 * np.pi * vab)
        ix3 = (x3b - x3a) / (4 * np.pi * vab)

        # Add contribution from wire pp into array
        hx0 = hx0 + phi * (-ix2 * rx3 + ix3 * rx2) / r
        hy0 = hy0 + phi * (ix1 * rx3 - ix3 * rx1) / r
        hz0 = hz0 + phi * (-ix1 * rx2 + ix2 * rx1) / r

    return [hx0, hy0, hz0]
