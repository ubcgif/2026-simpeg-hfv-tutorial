import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from discretize import TreeMesh, TensorMesh
from discretize.utils import mkvc
from simpeg.utils import ndgrid, validate_float, validate_integer, validate_string

TOL = 1e-10


def _define_float_properties(*prop_names):
    """A class decorator that generates float properties dynamically."""

    def decorator(cls):
        for name in prop_names:
            private_name = f"_{name}"

            str_name = name.replace("_", " ")
            prop_doc = f"""The '{str_name}' Maxwell plate parameter.

Returns
-------
float
    The '{str_name}' Maxwell plate parameter.
"""

            def getter(self, p_name=private_name):
                return getattr(self, p_name)

            def setter(self, value, p_name=private_name, public_name=name):
                value = validate_float(public_name, value)
                setattr(self, p_name, value)
                setattr(self, "_nodes", None)

            setattr(cls, name, property(getter, setter, doc=prop_doc))

        return cls

    return decorator

@_define_float_properties(
    "east",
    "north",
    "level",
    "dip",
    "dip_direction",
    "plunge",
    "length",
    "depth_extent",
    "conductance",
)
class MaxwellPlate:
    """Maxwell plate model object.

    This class stores the properties required to define a Maxwell plate.

    Parameters
    ----------
    name : str
        The name of the Maxwell plate.
    east : float
        The Easting location of the hinge point.
    north : float
        The Northing location of the hinge point.
    level : float
        The elevation of the hinge point.
    dip : float
        The dip angle for the plate [-90, 90].
    dip_direction : float
        The dip direction [0, 360], defined clockwise from North.
    plunge : float
        The plunge angle [0, 360] for the plate.
    length : float
        The physical strike distance measuring the horizontal span of
        the rectangular plate in meters.
    depth_extent : float
        The physical down-dip width measuring the downward slope distance
        of the plate in meters.
    conductance : float
        The conductivity-thickness product (conductance) of the plate.
    ribbons : int
        The number of current ribbons used to approximate the plate.
    """

    def __init__(
        self,
        name=None,
        east=None,
        north=None,
        level=None,
        dip=None,
        dip_direction=None,
        plunge=None,
        length=None,
        depth_extent=None,
        conductance=None,
        ribbons=20,
    ):
        self.name = name
        self.east = east
        self.north = north
        self.level = level
        self.dip = dip
        self.dip_direction = dip_direction
        self.plunge = plunge
        self.length = length
        self.depth_extent = depth_extent
        self.conductance = conductance
        self.ribbons = ribbons

    @property
    def name(self):
        """Plate name.

        Returns
        -------
        str
            Plate name.
        """
        return self._name

    @name.setter
    def name(self, val):
        self._name = validate_string(
            "name", val, string_list=None, case_sensitive=False
        )

    @property
    def ribbons(self):
        """Number of ribbons.

        Returns
        -------
        int
            Number of ribbons.
        """
        return self._ribbons

    @ribbons.setter
    def ribbons(self, val):
        self._ribbons = validate_integer("ribbons", val, min_val=1)

    @property
    def nodes(self):
        """Output node locations for the Maxwell plate.

        Returns
        -------
        (n, 4) numpy.ndarray
            The node locations of the plate.
        """

        #
        # x4----------------x3
        # |                 |
        # |                 |
        # |                 |
        # |                 |
        # |                 |
        # |                 |
        # x1-------O--------x2
        #     <--Length-->

        if getattr(self, "_nodes", None) is None:

            # Get x1 through x4 on local coordinate system (O is origin)
            # Length is the width along strike direction (W to E)
            # Depth extent is the width along dip direction (S to N)
            # We start with a plate defined on the XY plane. where x0 is bottom edge
            x1 = np.r_[-self.length / 2, 0.0, 0.0]
            x2 = np.r_[self.length / 2, 0.0, 0.0]
            x3 = np.r_[self.length / 2, self.depth_extent, 0.0]
            x4 = np.r_[-self.length / 2, self.depth_extent, 0.0]
            nodes = np.c_[x1, x2, x3, x4]

            dip = self.dip * np.pi / 180
            dip_direction = self.dip_direction * np.pi / 180
            plunge = self.plunge * np.pi / 180

            # Apply plunge rotation angle (rotation about z-axis)
            A1 = np.array(
                [
                    [np.cos(plunge), -np.sin(plunge), 0.0],
                    [np.sin(plunge), np.cos(plunge), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

            # Apply dip angle (rotation about x-axis)
            A2 = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(dip), np.sin(dip)],
                    [0.0, -np.sin(dip), np.cos(dip)],
                ]
            )

            # Apply dip direction (rotation about z-axis)
            A3 = np.array(
                [
                    [np.cos(dip_direction), np.sin(dip_direction), 0.0],
                    [-np.sin(dip_direction), np.cos(dip_direction), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

            nodes = A3 @ (A2 @ (A1 @ nodes))

            # Shift and return
            nodes = nodes.T + np.c_[self.east, self.north, self.level]

            self._nodes = nodes

        return self._nodes


def write_pte_file(fname, plate_list):
    """Writes list of Maxwell plates to PTE file.

    Parameters
    ----------
    fname : str
        Filename.
    plate_list : list of ``MaxwellPlate``
        List of Maxwell plate objects.
    """

    if not isinstance(plate_list, list):
        plate_list = [plate_list]
    for plate in plate_list:

        if not isinstance(plate, MaxwellPlate):
            raise TypeError(
                f"Expected a {MaxwellPlate.__name__}, but got {type(plate).__name__}"
            )

    with open(fname, "w") as f:
        f.write("MAXWELL PLATE FILE")
        f.write(f"\nTOTAL PLATES : {len(plate_list)}\n")

    for ii, plate in enumerate(plate_list):

        out_string = f"""****************************
PLATE         : {ii+1}
Name          : {plate.name}
East          : {plate.east}
North         : {plate.north}
Level         : {plate.level}
Dip           : {plate.dip}
Dip Direction : {plate.dip_direction}
Plunge        : {plate.plunge}
Length        : {plate.length}
Depth Extent  : {plate.depth_extent}
c-t           : {plate.conductance}
Ribbons       : {plate.ribbons}
        """

        with open(fname, "a") as f:
            f.write("\n" + out_string)


def read_pte_file(fname):
    """Loads PTE file into a list of Maxwell plates

    Parameters
    ----------
    fname : str
        The file path to the PTE file.

    Returns
    -------
    list of ``MaxwellPlate``
        Returns a list of Maxwell plate objects.
    """
    # Mappings from property name in file to property name in SimPEG
    key_mapping = {
        "Name": "name",
        "East": "east",
        "North": "north",
        "Level": "level",
        "Dip": "dip",
        "Dip Direction": "dip_direction",
        "Plunge": "plunge",
        "Length": "length",
        "Depth Extent": "depth_extent",
        "c-t": "conductance",
        "Ribbons": "ribbons",
    }

    plates_list = []
    current_plate_data = {}

    # Track whether the parser is currently reading a section we want to skip
    skip_remaining_plate_fields = False

    with open(fname, "r") as f:
        for line in f:
            line = line.strip()

            # Skip overall file headers or block separators
            if (
                not line
                or line.startswith("MAXWELL")
                or line.startswith("TOTAL")
                or line.startswith("*****")
            ):
                continue

            # When hitting inversion or styling blocks, stop capturing fields for the current plate
            if (
                line.startswith("INVERT ON PARAMETERS?")
                or line.startswith("COLE-COLE")
                or line.startswith("PLATE BORDER")
                or line.startswith("RIBBONS")
                or line.startswith("PLATE COLOUR")
            ):
                skip_remaining_plate_fields = True
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                # 'PLATE' signifies the start of a brand new plate block
                if key == "PLATE":
                    if current_plate_data:
                        plates_list.append(_build_plate_from_dict(current_plate_data))
                        current_plate_data = {}
                    # Reset the parsing filter for the new plate block
                    skip_remaining_plate_fields = False
                    continue

                # Only capture keys if we are in the primary parameters section of the plate
                if not skip_remaining_plate_fields and key in key_mapping:
                    current_plate_data[key_mapping[key]] = value

        # Catch the final plate block at the end of the file loop
        if current_plate_data:
            plates_list.append(_build_plate_from_dict(current_plate_data))

    return plates_list


def _build_plate_from_dict(data_dict):
    """Helper to convert parsed strings to proper python types before instantiation."""
    int_fields = {"ribbons"}
    str_fields = {"name"}

    clean_kwargs = {}
    for k, v in data_dict.items():
        if k in str_fields:
            clean_kwargs[k] = v
        elif k in int_fields:
            clean_kwargs[k] = int(v)
        else:
            clean_kwargs[k] = float(v)

    return MaxwellPlate(**clean_kwargs)


# def maxwell_plate_to_nodes(
#     east, north, level, dip, dip_direction, plunge, length, depth_extent
# ):
#     """
#     Obtain the nodes for a Maxwell plate.

#     Parameters
#     ----------
#     east : float
#         The easting or "x" location of the plate reference edge (PRE).
#     north : float
#         The northing or "y" location of the plate reference edge (PRE).
#     level : float
#         The elevation or "z" location of the plate reference edge (PRE).
#     dip : float
#         The dip angle in degrees. The dip angle of the vector normal to the plate.
#         A dip angle of 0 is a vertical plate.
#     dip_direction : float
#         The dip direction angle in degrees. Azimuthal angle of the down-dip
#         direction clockwise from North.
#     plunge : float
#         The plunge angle in degrees. Rotation about the vector normal to the plate.
#     length : float
#         Length along the strike direction.
#     depth_extent : float
#         Length along the dip direction.

#     Returns
#     -------
#     (n, 4) numpy.ndarray
#         The node locations of the plate.
#     """

#     #
#     # x4----------------x3
#     # |                 |
#     # |                 |
#     # |                 |
#     # |                 |
#     # |                 |
#     # |                 |
#     # x1-------O--------x2
#     #     <--Length-->

#     # Get x1 through x4 on local coordinate system (O is origin)
#     # Length is the width along strike direction (W to E)
#     # Depth extent is the width along dip direction (S to N)
#     # We start with a plate defined on the XY plane. where x0 is bottom edge
#     x1 = np.r_[-length / 2, 0.0, 0.0]
#     x2 = np.r_[length / 2, 0.0, 0.0]
#     x3 = np.r_[length / 2, depth_extent, 0.0]
#     x4 = np.r_[-length / 2, depth_extent, 0.0]
#     nodes = np.c_[x1, x2, x3, x4]

#     dip *= np.pi / 180
#     dip_direction *= np.pi / 180
#     plunge *= np.pi / 180

#     # Apply plunge rotation angle (rotation about z-axis)
#     A1 = np.array(
#         [
#             [np.cos(plunge), -np.sin(plunge), 0.0],
#             [np.sin(plunge), np.cos(plunge), 0.0],
#             [0.0, 0.0, 1.0],
#         ]
#     )

#     # Apply dip angle (rotation about x-axis)
#     A2 = np.array(
#         [
#             [1.0, 0.0, 0.0],
#             [0.0, np.cos(dip), np.sin(dip)],
#             [0.0, -np.sin(dip), np.cos(dip)],
#         ]
#     )

#     # Apply dip direction (rotation about z-axis)
#     A3 = np.array(
#         [
#             [np.cos(dip_direction), np.sin(dip_direction), 0.0],
#             [-np.sin(dip_direction), np.cos(dip_direction), 0.0],
#             [0.0, 0.0, 1.0],
#         ]
#     )

#     nodes = A3 @ (A2 @ (A1 @ nodes))

#     # Shift and return
#     return nodes.T + np.c_[east, north, level]


def dipping_plate_to_nodes(
    center, strike_width, dip_width, strike_angle, dip_angle, plunge_angle=0.0
):
    """Obtain nodes for a dipping plate.

    Parameters
    ----------
    center : numpy.ndarray (3,)
        The center of the plate in Cartesian coordinates.
    strike_width : float
        Sets the width along the first plate-reference axis direction of the plate.
        For a plunge angle of 0, this is equal to the width along the strike direction.
    dip_width : float
        Sets the width along the second plate-reference axis direction of the plate.
        For a plunge angle of 0, this is equal to the width perpendicular to the strike direction.
    strike_angle : float
        Strike angle in degrees, ``[0, 360]``.
    dip_angle : float
        Dip angle in degrees, ``[-90, 90]``.
    plunge_angle : float
        Rotation angle about the axis normal to the plate in degrees, [``-180, 180``].

    Returns
    -------
    (n, 4) numpy.ndarray
        The node locations of the plate.
    """

    # Apply rotations about center reference point
    plunge_angle *= np.pi / 180
    dip_angle *= np.pi / 180
    strike_angle *= np.pi / 180

    # Construct rotation matrices
    A1 = np.array(
        [
            [np.cos(plunge_angle), np.sin(plunge_angle), 0.0],
            [-np.sin(plunge_angle), np.cos(plunge_angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    A2 = np.array(
        [
            [np.cos(dip_angle), 0.0, np.sin(dip_angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(dip_angle), 0.0, np.cos(dip_angle)],
        ]
    )

    A3 = np.array(
        [
            [np.cos(strike_angle), np.sin(strike_angle), 0.0],
            [-np.sin(strike_angle), np.cos(strike_angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Define reference rectangle
    pts = np.array(
        [
            [-strike_width / 2, strike_width / 2, strike_width / 2, -strike_width / 2],
            [-strike_width / 2, -strike_width / 2, strike_width / 2, strike_width / 2],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    pts = np.dot(A3, np.dot(A2, np.dot(A1, pts)))

    return pts.T + center.reshape((1, 3))


def _get_subdomain_cell_indices(mesh, points, simplices):
    """Get cells defining subdomain."""

    # Build kdtree from triangle centers
    kdtree = cKDTree(np.mean(points[simplices], axis=1))

    # Query neighbors within specified radius.
    # Radius based on largest edge length for triangle
    radius = points[simplices] - points[simplices[:, [1, 2, 0]]]
    radius = np.sqrt(np.sum(radius**2, axis=2))
    radius = np.sqrt(3) * np.max(radius)
    indices_inside_ball = kdtree.query_ball_point(mesh.cell_centers, r=radius)

    # Return the boolean mask
    return np.array([len(neighbors) > 0 for neighbors in indices_inside_ball])


def _get_lower_upper_faces(mesh, subdom_inds=None):

    if isinstance(mesh, TreeMesh):

        cells = mesh[:]
        if subdom_inds is not None:
            cells = [cells[i] for i in np.where(subdom_inds)[0]]
        sum_n_cells = [0, mesh.n_faces_x, mesh.n_faces_x + mesh.n_faces_y]

        face_ind_by_cell = [
            np.array([c.faces[1 + 2 * ii] for c in cells], dtype=int) for ii in range(3)
        ]
        inds = [
            face_ind_by_cell[ii] < mesh.n_faces_per_direction[ii] for ii in range(3)
        ]
        if subdom_inds is None:
            dh_0 = np.hstack([mesh.h_gridded[inds[ii], ii] for ii in range(3)])
        else:
            dh_0 = np.hstack(
                [mesh.h_gridded[subdom_inds][inds[ii], ii] for ii in range(3)]
            )
        face_ind_by_cell = [x[inds[ii]] for ii, x in enumerate(face_ind_by_cell)]
        face_inds_0 = np.hstack(
            [face_ind_by_cell[ii] + sum_n_cells[ii] for ii in range(3)]
        )

        face_ind_by_cell = [
            np.array([c.faces[2 * ii] for c in cells], dtype=int) for ii in range(3)
        ]
        inds = [
            face_ind_by_cell[ii] < mesh.n_faces_per_direction[ii] for ii in range(3)
        ]
        if subdom_inds is None:
            dh_1 = np.hstack([mesh.h_gridded[inds[ii], ii] for ii in range(3)])
        else:
            dh_1 = np.hstack(
                [mesh.h_gridded[subdom_inds][inds[ii], ii] for ii in range(3)]
            )
        face_ind_by_cell = [x[inds[ii]] for ii, x in enumerate(face_ind_by_cell)]
        face_inds_1 = np.hstack(
            [face_ind_by_cell[ii] + sum_n_cells[ii] for ii in range(3)]
        )

    elif isinstance(mesh, TensorMesh):

        if subdom_inds is None:
            bf_inds = mesh.face_boundary_indices
            face_inds_0 = np.hstack([~bf_inds[2 * ii] for ii in range(3)])
            face_inds_1 = np.hstack([~bf_inds[1 + 2 * ii] for ii in range(3)])
            dh_0 = dh_1 = mkvc(mesh.h_gridded)
        else:

            ncx, ncy, ncz = mesh.shape_cells
            ijk = ndgrid(np.arange(ncx), np.arange(ncy), np.arange(ncz))
            ijk = ijk[subdom_inds]

            face_inds_0 = [
                ijk[:, 0] + (ncx + 1) * ijk[:, 1] + (ncx + 1) * ncy * ijk[:, 2],
                ijk[:, 0] + ncx * ijk[:, 1] + ncx * (ncy + 1) * ijk[:, 2],
                ijk[:, 0] + ncx * ijk[:, 1] + ncx * ncy * ijk[:, 2],
            ]
            face_inds_1 = [
                (ijk[:, 0] + 1) + (ncx + 1) * ijk[:, 1] + (ncx + 1) * ncy * ijk[:, 2],
                ijk[:, 0] + ncx * (ijk[:, 1] + 1) + ncx * (ncy + 1) * ijk[:, 2],
                ijk[:, 0] + ncx * ijk[:, 1] + ncx * ncy * (ijk[:, 2] + 1),
            ]
            temp = [0, mesh.n_faces_x, mesh.n_faces_x + mesh.n_faces_y]

            face_inds_0 = np.hstack([a + b for a, b in zip(temp, face_inds_0)])
            face_inds_1 = np.hstack([a + b for a, b in zip(temp, face_inds_1)])

            dh_0 = dh_1 = mkvc(mesh.h_gridded[subdom_inds, :])

    return face_inds_0, face_inds_1, dh_0, dh_1


def _get_plane_distance_condition(
    pts, faces, face_normals, face_inds_0, face_inds_1, dh_0, dh_1, tol=1e-10
):

    # Plane on which our plate lives
    nvec = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    nvec /= np.linalg.norm(nvec)
    c = np.dot(nvec, pts[0])

    # Scaled distance from triangle conditions
    n_faces = len(faces)
    condition_0 = np.zeros(n_faces, dtype=bool)
    condition_1 = np.zeros(n_faces, dtype=bool)

    faces_dot_nvec = np.dot(faces, nvec)
    normals_dot_nvec = np.dot(face_normals, nvec)
    normals_dot_nvec *= np.sign(normals_dot_nvec)

    condition_0[face_inds_0] = (
        faces_dot_nvec[face_inds_0] - normals_dot_nvec[face_inds_0] * dh_0 / 2.0 - c
        <= TOL
    )
    condition_1[face_inds_1] = (
        faces_dot_nvec[face_inds_1] + normals_dot_nvec[face_inds_1] * dh_1 / 2.0 - c
        > TOL
    )
    condition_2 = np.abs(np.dot(face_normals, nvec)) >= TOL

    return condition_0 & condition_1 & condition_2


def _get_confining_condition_rectangle(pts, faces, tol=1e-10):

    # Define bounding plane equations.
    nvec01 = (pts[1] - pts[0]) / np.linalg.norm(pts[1] - pts[0])
    nvec12 = (pts[2] - pts[1]) / np.linalg.norm(pts[2] - pts[1])
    nvec23 = (pts[3] - pts[2]) / np.linalg.norm(pts[3] - pts[2])
    nvec30 = (pts[0] - pts[3]) / np.linalg.norm(pts[0] - pts[3])

    c01 = np.sum(nvec01 * pts[0])
    c12 = np.sum(nvec12 * pts[1])
    c23 = np.sum(nvec23 * pts[2])
    c30 = np.sum(nvec30 * pts[3])

    return (
        (np.dot(faces, nvec01) - c01 >= TOL)
        & (np.dot(faces, nvec12) - c12 >= TOL)
        & (np.dot(faces, nvec23) - c23 >= TOL)
        & (np.dot(faces, nvec30) - c30 >= TOL)
    )


def get_face_indices_plate(mesh, nodes):
    """Face indices for plate projected to mesh.

    Returns the indices of the mesh faces that discretize a rectangular plate onto a
    a mesh. Used to define models for physical properties on mesh faces.

    **Please note** this function is not robust if the cell dimensions change over
    the surface of the plate.


    Parameters
    ----------
    mesh : discretize.TensorMesh or discretize.TreeMesh
        The mesh. Must be a 3D mesh.
    nodes : list of (4, 3) np.ndarray
        List of plate nodes.

    Returns
    -------
    numpy.ndarray of int
        Face indices for plate projected to mesh.

    """

    if mesh.dim < 3:
        raise TypeError(
            "'get_face_indices_plate' requires a 3D mesh. Your mesh is {}D.".format(
                mesh.dim
            )
        )

    if not isinstance(nodes, list):
        nodes = [nodes]

    # SETUP
    faces = mesh.faces
    face_normals = mesh.face_normals

    # Properly oriented faces within scaled distance from each triangle
    face_inds_0, face_inds_1, dh_0, dh_1 = _get_lower_upper_faces(mesh)

    k = np.zeros(mesh.n_faces, dtype=bool)
    for pts in nodes:

        # Scaled distance from plane condition
        condition_distance = _get_plane_distance_condition(
            pts, faces, face_normals, face_inds_0, face_inds_1, dh_0, dh_1, tol=TOL
        )

        # Confining planes condition
        condition_confining = _get_confining_condition_rectangle(pts, faces, tol=TOL)

        k |= condition_distance & condition_confining

    return k


def get_face_projection_plate(mesh, nodes):
    """Face indices for plate projected to mesh.

    Returns the indices of the mesh faces that discretize a rectangular plate onto a
    a mesh. Used to define models for physical properties on mesh faces.

    **Please note** this function is not robust if the cell dimensions change over
    the surface of the plate.


    Parameters
    ----------
    mesh : discretize.TensorMesh or discretize.TreeMesh
        The mesh. Must be a 3D mesh.
    nodes : list of (4, 3) np.ndarray
        List of plate nodes.

    Returns
    -------
    numpy.ndarray of int
        Face indices for plate projected to mesh.

    """

    if mesh.dim < 3:
        raise TypeError(
            "'get_face_indices_plate' requires a 3D mesh. Your mesh is {}D.".format(
                mesh.dim
            )
        )

    if not isinstance(nodes, list):
        nodes = [nodes]
    n_plates = len(nodes)

    # SETUP
    faces = mesh.faces
    face_normals = mesh.face_normals

    # Properly oriented faces within scaled distance from each triangle
    face_inds_0, face_inds_1, dh_0, dh_1 = _get_lower_upper_faces(mesh)

    k = np.zeros(mesh.n_faces, dtype=bool)
    P = sp.lil_matrix((mesh.n_faces, n_plates))
    for nn, pts in enumerate(nodes):

        # Scaled distance from plane condition
        condition_distance = _get_plane_distance_condition(
            pts, faces, face_normals, face_inds_0, face_inds_1, dh_0, dh_1, tol=TOL
        )

        # Confining planes condition
        condition_confining = _get_confining_condition_rectangle(pts, faces, tol=TOL)

        k = np.where((condition_distance & condition_confining))[0]

        P[k, nn] = 1.0

    sum_rows = np.sum(P, axis=1)
    temp = np.divide(
        np.ones((mesh.n_faces, 1)),
        sum_rows,
        out=np.zeros((mesh.n_faces, 1)),
        where=sum_rows != 0.0,
    )

    return (P.multiply(temp)).tocsr()


def get_face_indices_simplex(mesh, points, simplices, force_orientation=False):
    """Face indices for surface defined by a 2D simplex.

    Returns the indices of the mesh faces that discretize a rectangular plate onto a
    a mesh. Used to define models for physical properties on mesh faces.

    **Please note** this function is not robust if the cell dimensions change over
    the surface of the plate.


    Parameters
    ----------
    mesh : discretize.TensorMesh or discretize.TreeMesh
        The mesh. Must be a 3D mesh.
    points : (n, 3) numpy.ndarray
        Points in 3D space.
    simplices : (k, 3) numpy.ndarray of int
        Simplices.

    Returns
    -------
    numpy.ndarray of int
        Face indices for simplex plate projected to mesh.

    """
    # Checks
    if mesh.dim < 3:
        raise TypeError(
            "'get_face_indices_plate' requires a 3D mesh. Your mesh is {}D.".format(
                mesh.dim
            )
        )

    # Rare instances of missing faces when points lie directly on mesh geometry (shift by 1e-6)
    points += 1e-6

    # NEED SOMETHING TO ENSURE SIMPLEXES ARE ORDERED CCW!!!

    n_triangles = len(simplices)

    # FIND MESH SUBDOMAIN (avoids redundant computation for faces away from surface)
    subdom_inds = _get_subdomain_cell_indices(mesh, points, simplices)
    face_inds_0, face_inds_1, dh_0, dh_1 = _get_lower_upper_faces(mesh, subdom_inds)

    # FACE INDS FOR SUBDOMAIN
    face_inds_subdom = np.unique(np.r_[face_inds_0, face_inds_1])
    sort_idx = np.argsort(face_inds_subdom)
    face_inds_0 = sort_idx[
        np.searchsorted(face_inds_subdom, face_inds_0, sorter=sort_idx)
    ]
    face_inds_1 = sort_idx[
        np.searchsorted(face_inds_subdom, face_inds_1, sorter=sort_idx)
    ]

    faces = mesh.faces[face_inds_subdom, :]
    face_normals = mesh.face_normals[face_inds_subdom, :]

    # Pre-allocate
    k = np.zeros(len(faces), dtype=bool)
    one_minus_face_normals = 1.0 - face_normals

    for nn in range(n_triangles):

        pts = [points[ind, :] for ind in simplices[nn, :]]

        # Scaled distance from plane condition
        condition_distance = _get_plane_distance_condition(
            pts, faces, face_normals, face_inds_0, face_inds_1, dh_0, dh_1, tol=TOL
        )

        # # Confining planes condition

        # WOULD LIKE TO CALL A FUNCTION, HOWEVER LARGE ARRAYS GENERATED AT EACH ITERATION.
        # condition_confining = _get_confining_condition_triangle(
        #     pts, faces, face_normals, sig_fig=sig_fig
        # )

        ppt0 = pts[0] * one_minus_face_normals
        ppt1 = pts[1] * one_minus_face_normals
        ppt2 = pts[2] * one_minus_face_normals

        # Perpendicular points
        num01 = (ppt1 - ppt0) * np.sum((ppt2 - ppt0) * (ppt1 - ppt0), axis=1)[:, None]
        num12 = (ppt2 - ppt1) * np.sum((ppt0 - ppt1) * (ppt2 - ppt1), axis=1)[:, None]
        num20 = (ppt0 - ppt2) * np.sum((ppt1 - ppt2) * (ppt0 - ppt2), axis=1)[:, None]

        den01 = np.sum((ppt1 - ppt0) ** 2, axis=1)[:, None]
        den12 = np.sum((ppt2 - ppt1) ** 2, axis=1)[:, None]
        den20 = np.sum((ppt0 - ppt2) ** 2, axis=1)[:, None]

        perp01 = ppt0 + np.divide(
            num01, den01, out=np.zeros_like(num01), where=den01 != 0
        )
        perp12 = ppt1 + np.divide(
            num12, den12, out=np.zeros_like(num12), where=den12 != 0
        )
        perp20 = ppt2 + np.divide(
            num20, den20, out=np.zeros_like(num20), where=den20 != 0
        )

        # Confining planes
        den01 = np.sum((ppt2 - perp01) ** 2, axis=1)[:, None]
        den12 = np.sum((ppt0 - perp12) ** 2, axis=1)[:, None]
        den20 = np.sum((ppt1 - perp20) ** 2, axis=1)[:, None]

        nvec01 = np.divide(
            ppt2 - perp01, den01, out=np.zeros_like(ppt2 - perp01), where=den01 != 0
        )
        nvec12 = np.divide(
            ppt0 - perp12, den12, out=np.zeros_like(ppt0 - perp12), where=den12 != 0
        )
        nvec20 = np.divide(
            ppt1 - perp20, den20, out=np.zeros_like(ppt1 - perp20), where=den20 != 0
        )

        c01 = np.sum(nvec01 * ppt0, axis=1)
        c12 = np.sum(nvec12 * ppt1, axis=1)
        c20 = np.sum(nvec20 * ppt2, axis=1)

        condition_confining = (
            (np.sum(faces * nvec01, axis=1) - c01 >= -TOL)
            & (np.sum(faces * nvec12, axis=1) - c12 >= -TOL)
            & (np.sum(faces * nvec20, axis=1) - c20 >= -TOL)
        )

        k |= condition_distance & condition_confining

    global_face_inds = np.zeros(mesh.n_faces, dtype=bool)
    global_face_inds[face_inds_subdom] = k

    return global_face_inds


def get_face_projection_simplex(mesh, points, simplices, force_orientation=False):
    """Face indices for surface defined by a 2D simplex.

    Returns the indices of the mesh faces that discretize a rectangular plate onto a
    a mesh. Used to define models for physical properties on mesh faces.

    **Please note** this function is not robust if the cell dimensions change over
    the surface of the plate.


    Parameters
    ----------
    mesh : discretize.TensorMesh or discretize.TreeMesh
        The mesh. Must be a 3D mesh.
    points : (n, 3) numpy.ndarray
        Points in 3D space.
    simplices : (k, 3) numpy.ndarray of int
        Simplices.

    Returns
    -------
    numpy.ndarray of int
        Face indices for simplex plate projected to mesh.

    """
    # Checks
    if mesh.dim < 3:
        raise TypeError(
            "'get_face_indices_plate' requires a 3D mesh. Your mesh is {}D.".format(
                mesh.dim
            )
        )

    # if ~isinstance(mesh, (TensorMesh, TreeMesh)):
    #     raise NotImplementedError("'get_face_indices_simplex' not implemented for mesh class {}.".format(type(mesh)))

    # Rare instances of missing faces when points lie directly on mesh geometry (shift by 1e-6)
    points += 1e-6

    # NEED SOMETHING TO ENSURE SIMPLEXES ARE ORDERED CCW!!!

    # GLOBAL PROPERTIES
    n_triangles = len(simplices)

    # FIND MESH SUBDOMAIN (avoids redundant computation for faces away from surface)
    subdom_inds = _get_subdomain_cell_indices(mesh, points, simplices)
    face_inds_0, face_inds_1, dh_0, dh_1 = _get_lower_upper_faces(mesh, subdom_inds)

    face_inds_subdom = np.unique(np.r_[face_inds_0, face_inds_1])
    sort_idx = np.argsort(face_inds_subdom)
    face_inds_0 = sort_idx[
        np.searchsorted(face_inds_subdom, face_inds_0, sorter=sort_idx)
    ]
    face_inds_1 = sort_idx[
        np.searchsorted(face_inds_subdom, face_inds_1, sorter=sort_idx)
    ]

    faces = mesh.faces[face_inds_subdom, :]
    face_normals = mesh.face_normals[face_inds_subdom, :]

    # Pre-allocate
    k = np.zeros(mesh.n_faces, dtype=bool)
    P = sp.lil_matrix((mesh.n_faces, n_triangles))
    one_minus_face_normals = 1.0 - face_normals

    for nn in range(n_triangles):

        pts = [points[ind, :] for ind in simplices[nn, :]]

        # Scaled distance from plane condition
        condition_distance = _get_plane_distance_condition(
            pts, faces, face_normals, face_inds_0, face_inds_1, dh_0, dh_1, tol=TOL
        )

        # # Confining planes condition

        # WOULD LIKE TO CALL A FUNCTION, HOWEVER LARGE ARRAYS GENERATED AT EACH ITERATION.
        # condition_confining = _get_confining_condition_triangle(
        #     pts, faces, face_normals, sig_fig=sig_fig
        # )

        # Projected points
        ppt0 = pts[0] * one_minus_face_normals
        ppt1 = pts[1] * one_minus_face_normals
        ppt2 = pts[2] * one_minus_face_normals

        # Perpendicular points
        num01 = (ppt1 - ppt0) * np.sum((ppt2 - ppt0) * (ppt1 - ppt0), axis=1)[:, None]
        num12 = (ppt2 - ppt1) * np.sum((ppt0 - ppt1) * (ppt2 - ppt1), axis=1)[:, None]
        num20 = (ppt0 - ppt2) * np.sum((ppt1 - ppt2) * (ppt0 - ppt2), axis=1)[:, None]

        den01 = np.sum((ppt1 - ppt0) ** 2, axis=1)[:, None]
        den12 = np.sum((ppt2 - ppt1) ** 2, axis=1)[:, None]
        den20 = np.sum((ppt0 - ppt2) ** 2, axis=1)[:, None]

        perp01 = ppt0 + np.divide(
            num01, den01, out=np.zeros_like(num01), where=den01 != 0
        )
        perp12 = ppt1 + np.divide(
            num12, den12, out=np.zeros_like(num12), where=den12 != 0
        )
        perp20 = ppt2 + np.divide(
            num20, den20, out=np.zeros_like(num20), where=den20 != 0
        )

        # Confining planes
        den01 = np.sum((ppt2 - perp01) ** 2, axis=1)[:, None]
        den12 = np.sum((ppt0 - perp12) ** 2, axis=1)[:, None]
        den20 = np.sum((ppt1 - perp20) ** 2, axis=1)[:, None]

        nvec01 = np.divide(
            ppt2 - perp01, den01, out=np.zeros_like(ppt2 - perp01), where=den01 != 0
        )
        nvec12 = np.divide(
            ppt0 - perp12, den12, out=np.zeros_like(ppt0 - perp12), where=den12 != 0
        )
        nvec20 = np.divide(
            ppt1 - perp20, den20, out=np.zeros_like(ppt1 - perp20), where=den20 != 0
        )

        c01 = np.sum(nvec01 * ppt0, axis=1)
        c12 = np.sum(nvec12 * ppt1, axis=1)
        c20 = np.sum(nvec20 * ppt2, axis=1)

        condition_confining = (
            (np.sum(faces * nvec01, axis=1) - c01 >= -TOL)
            & (np.sum(faces * nvec12, axis=1) - c12 >= -TOL)
            & (np.sum(faces * nvec20, axis=1) - c20 >= -TOL)
        )

        k = np.zeros(mesh.n_faces, dtype=bool)
        k[face_inds_subdom] = condition_distance & condition_confining

        rows = np.where(k)[0]

        P[rows, nn] = 1.0

    sum_rows = np.sum(P, axis=1)
    temp = np.divide(
        np.ones((mesh.n_faces, 1)),
        sum_rows,
        out=np.zeros((mesh.n_faces, 1)),
        where=sum_rows != 0.0,
    )

    return (P.multiply(temp)).tocsr()
