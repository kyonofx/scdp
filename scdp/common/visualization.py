from ase.data import chemical_symbols, covalent_radii
import numpy as np
import plotly.graph_objects as go

CPK_RGB = [
    "rgb(0, 0, 0)",
    "rgb(223, 223, 223)", # make H more visible under white background
    "rgb(217, 255, 255)",
    "rgb(204, 128, 255)",
    "rgb(194, 255, 0)",
    "rgb(255, 181, 181)",
    "rgb(144, 144, 144)",
    "rgb(48, 80, 248)",
    "rgb(255, 13, 13)",
    "rgb(144, 224, 80)",
    "rgb(179, 227, 245)",
    "rgb(171, 92, 242)",
    "rgb(138, 255, 0)",
    "rgb(191, 166, 166)",
    "rgb(240, 200, 160)",
    "rgb(255, 128, 0)",
    "rgb(255, 255, 48)",
    "rgb(31, 240, 31)",
    "rgb(128, 209, 227)",
    "rgb(143, 64, 212)",
    "rgb(61, 255, 0)",
    "rgb(230, 230, 230)",
    "rgb(191, 194, 199)",
    "rgb(166, 166, 171)",
    "rgb(138, 153, 199)",
    "rgb(156, 122, 199)",
    "rgb(224, 102, 51)",
    "rgb(240, 144, 160)",
    "rgb(80, 208, 80)",
    "rgb(200, 128, 51)",
    "rgb(125, 128, 176)",
    "rgb(194, 143, 143)",
    "rgb(102, 143, 143)",
    "rgb(189, 128, 227)",
    "rgb(255, 161, 0)",
    "rgb(166, 41, 41)",
    "rgb(92, 184, 209)",
    "rgb(112, 46, 176)",
    "rgb(0, 255, 0)",
    "rgb(148, 255, 255)",
    "rgb(148, 224, 224)",
    "rgb(115, 194, 201)",
    "rgb(84, 181, 181)",
    "rgb(59, 158, 158)",
    "rgb(36, 143, 143)",
    "rgb(10, 125, 140)",
    "rgb(0, 105, 133)",
    "rgb(192, 192, 192)",
    "rgb(255, 217, 143)",
    "rgb(166, 117, 115)",
    "rgb(102, 128, 128)",
    "rgb(158, 99, 181)",
    "rgb(212, 122, 0)",
    "rgb(148, 0, 148)",
    "rgb(66, 158, 176)",
    "rgb(87, 23, 143)",
    "rgb(0, 201, 0)",
    "rgb(112, 212, 255)",
    "rgb(255, 255, 199)",
    "rgb(217, 255, 199)",
    "rgb(199, 255, 199)",
    "rgb(163, 255, 199)",
    "rgb(143, 255, 199)",
    "rgb(97, 255, 199)",
    "rgb(69, 255, 199)",
    "rgb(48, 255, 199)",
    "rgb(31, 255, 199)",
    "rgb(0, 255, 156)",
    "rgb(0, 230, 117)",
    "rgb(0, 212, 82)",
    "rgb(0, 191, 56)",
    "rgb(0, 171, 36)",
    "rgb(77, 194, 255)",
    "rgb(77, 166, 255)",
    "rgb(33, 148, 214)",
    "rgb(38, 125, 171)",
    "rgb(38, 102, 150)",
    "rgb(23, 84, 135)",
    "rgb(208, 208, 224)",
    "rgb(255, 209, 35)",
    "rgb(184, 184, 208)",
    "rgb(166, 84, 77)",
    "rgb(87, 89, 97)",
    "rgb(158, 79, 181)",
    "rgb(171, 92, 0)",
    "rgb(117, 79, 69)",
    "rgb(66, 130, 150)",
    "rgb(66, 0, 102)",
    "rgb(0, 125, 0)",
    "rgb(112, 171, 250)",
    "rgb(0, 186, 255)",
    "rgb(0, 161, 255)",
    "rgb(0, 143, 255)",
    "rgb(0, 128, 255)",
    "rgb(0, 107, 255)",
    "rgb(84, 92, 242)",
    "rgb(120, 92, 227)",
    "rgb(138, 79, 227)",
    "rgb(161, 54, 212)",
    "rgb(179, 31, 212)",
    "rgb(179, 31, 186)",
    "rgb(179, 13, 166)",
    "rgb(189, 13, 135)",
    "rgb(199, 0, 102)",
    "rgb(204, 0, 89)",
    "rgb(209, 0, 79)",
    "rgb(217, 0, 69)",
    "rgb(224, 0, 56)",
    "rgb(230, 0, 46)",
    "rgb(235, 0, 38)",
]


def lattice_object(cell, origin):
    if origin is None:
        origin = np.zeros(3)
    # Extract lattice vectors
    vec_a = cell[0]
    vec_b = cell[1]
    vec_c = cell[2]

    # Calculate points (vertices) of the unit cell
    point_b = origin + vec_a
    point_c = origin + vec_b
    point_d = origin + vec_c
    point_e = origin + vec_a + vec_b
    point_f = origin + vec_a + vec_c
    point_g = origin + vec_b + vec_c
    point_h = origin + vec_a + vec_b + vec_c

    # Define edges of the unit cell
    edges = [
        {"start": origin, "end": point_b},
        {"start": origin, "end": point_c},
        {"start": origin, "end": point_d},
        {"start": point_b, "end": point_e},
        {"start": point_b, "end": point_f},
        {"start": point_c, "end": point_e},
        {"start": point_c, "end": point_g},
        {"start": point_d, "end": point_f},
        {"start": point_d, "end": point_g},
        {"start": point_e, "end": point_h},
        {"start": point_f, "end": point_h},
        {"start": point_g, "end": point_h},
    ]

    # Create edge traces for plotting
    edge_traces = []
    for edge in edges:
        edge_trace = go.Scatter3d(
            x=[edge["start"][0], edge["end"][0]],
            y=[edge["start"][1], edge["end"][1]],
            z=[edge["start"][2], edge["end"][2]],
            mode="lines",
            line=dict(color="black", width=2),
        )
        edge_traces.append(edge_trace)

    return edge_traces


def draw_volume(
    grid_pos,
    density,
    atom_types,
    atom_coord,
    cell=None,
    origin=None,
    isomin=0.05,
    isomax=None,
    surface_count=5,
    downsample=4,
    title=None,
    dtype="volume",
):
    fig = go.Figure()

    if grid_pos is not None:
        grid_pos = grid_pos[::downsample, ::downsample, ::downsample]
    if density is not None:
        density = density[::downsample, ::downsample, ::downsample]
        
    if dtype == "volume":
        dobj = go.Volume(
            x=grid_pos[..., 0].flatten(),
            y=grid_pos[..., 1].flatten(),
            z=grid_pos[..., 2].flatten(),
            value=density.flatten(),
            isomin=isomin,
            isomax=isomax,
            opacity=0.1,  # needs to be small to see through all surfaces
            surface_count=surface_count,  # needs to be a large number for good volume rendering
            caps=dict(x_show=False, y_show=False, z_show=False),
            hoverinfo="skip",
        )
        fig.add_trace(dobj)
    elif dtype == "scatter":
        mask = density.flatten() > isomin
        dobj = go.Scatter3d(
            x=grid_pos[..., 0].flatten()[mask],
            y=grid_pos[..., 1].flatten()[mask],
            z=grid_pos[..., 2].flatten()[mask],
            mode="markers",
            marker=dict(
                size=3,
                color=density.flatten()[mask],
                opacity=0.1,
                cmin=isomin,
                cmax=isomax,
            ),
            hoverinfo="skip",
        )
        fig.add_trace(dobj)
    else:
        print("Charge density not rendered. To render it, specify <dtype> as <volume> or <scatter>.")

    axis_dict = dict(
        showgrid=False,
        showbackground=False,
        zeroline=False,
        visible=False,
    )

    fig.add_trace(
        go.Scatter3d(
            x=atom_coord[:, 0],
            y=atom_coord[:, 1],
            z=atom_coord[:, 2],
            mode="markers",
            text=[chemical_symbols[a] for a in atom_types],
            marker=dict(
                size=[20 * covalent_radii[a] for a in atom_types],
                color=[CPK_RGB[a] for a in atom_types],
                opacity=0.6,
            ),
        )
    )

    if cell is not None:
        fig.add_traces(lattice_object(cell, origin))

    if title is not None:
        title = dict(
            text=title,
            x=0.5,
            y=0.3,
            xanchor="center",
            yanchor="bottom",
        )
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        showlegend=False,
        scene=dict(xaxis=axis_dict, yaxis=axis_dict, zaxis=axis_dict),
        title=title,
        title_font_family="Times New Roman",
    )
    return fig
