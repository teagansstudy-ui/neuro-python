from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from nilearn import datasets, surface


st.set_page_config(
    page_title="Neuro Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKGROUND = "#10151d"
PANEL = "#171e28"
TEXT = "#eef1f5"
MUTED = "#9ca8b8"
ACCENT = "#71d7cf"

REGION_LABEL_POSITIONS = {
    "frontal": (48, 47, 48),
    "parietal": (48, -28, 63),
    "temporal": (67, -16, -7),
    "occipital": (38, -88, 21),
    "insula": (51, 13, 17),
    "hippocampus": (54, -26, -30),
    "amygdala": (51, 3, -22),
    "basal-ganglia": (31, 18, 26),
    "thalamus": (4, -19, 24),
    "hypothalamus": (4, 17, -2),
    "cerebellum": (-7, -83, -36),
    "pons": (-20, -30, -34),
    "medulla": (-18, -44, -54),
    "spinal-cord": (-12, -47, -83),
}

REGIONS = [
    {
        "id": "frontal",
        "name": "Frontal lobe",
        "position": (34, 36, 35),
        "description": "Planning, decision-making, voluntary movement, speech production, and aspects of personality.",
        "conditions": [
            "Stroke or a brain tumor can affect movement, speech, or planning in this region.",
            "Frontotemporal dementia commonly affects frontal and/or temporal networks.",
            "Infections such as encephalitis can affect the frontal lobe as part of a wider brain illness.",
        ],
    },
    {
        "id": "parietal",
        "name": "Parietal lobe",
        "position": (39, -28, 49),
        "description": "Combines touch and body-position information and supports spatial attention.",
        "conditions": [
            "Stroke, tumors, or multiple-sclerosis lesions may affect sensation or spatial processing.",
            "Encephalitis can involve parietal tissue, usually as part of a broader infection.",
            "Some inherited small-vessel conditions, including CADASIL, can cause brain lesions in multiple regions.",
        ],
    },
    {
        "id": "temporal",
        "name": "Temporal lobe",
        "position": (54, -15, 2),
        "description": "Supports hearing, language comprehension, memory, and recognition.",
        "conditions": [
            "Temporal-lobe epilepsy can cause recurrent seizures.",
            "Herpes simplex encephalitis often involves temporal and limbic regions.",
            "Rare inherited LGI1 variants can cause autosomal-dominant lateral temporal epilepsy.",
        ],
    },
    {
        "id": "occipital",
        "name": "Occipital lobe",
        "position": (30, -77, 14),
        "description": "The brain's main cortical area for processing visual information.",
        "conditions": [
            "Stroke or a tumor in visual cortex can cause visual-field changes.",
            "Infections or inflammation can affect this lobe as part of encephalitis.",
            "Inherited metabolic disorders such as MELAS may cause lesions in different brain regions, sometimes including occipital cortex.",
        ],
    },
    {
        "id": "insula",
        "name": "Insula",
        "position": (38, 0, 6),
        "description": "A deeper cortical area involved in taste, internal body sensations, emotion, and pain.",
        "conditions": [
            "Focal seizures and uncommon gliomas can arise in or involve the insula.",
            "Encephalitis may involve insular tissue, but is not limited to this area.",
            "There is no single common inherited disorder confined to the insula; genetic conditions can affect it as part of wider networks.",
        ],
    },
    {
        "id": "hippocampus",
        "name": "Hippocampus",
        "position": (25, -22, -12),
        "description": "A medial temporal structure important for forming new memories and spatial learning.",
        "conditions": [
            "Hippocampal sclerosis is associated with some forms of temporal-lobe epilepsy.",
            "Herpes simplex encephalitis can injure the hippocampus.",
            "Familial Alzheimer disease is inherited and can affect memory networks that include the hippocampus.",
        ],
    },
    {
        "id": "amygdala",
        "name": "Amygdala",
        "position": (24, -6, -12),
        "description": "Helps process emotional salience, threat, and emotional learning.",
        "conditions": [
            "Seizures and limbic encephalitis can involve the amygdala.",
            "Herpes simplex encephalitis can affect this structure along with nearby temporal regions.",
            "Urbach-Wiethe disease is a very rare inherited condition that may calcify the amygdala.",
        ],
    },
    {
        "id": "basal-ganglia",
        "name": "Basal ganglia",
        "position": (18, 3, 10),
        "description": "A group of deep nuclei involved in movement, habits, and action selection.",
        "conditions": [
            "Parkinson disease affects movement circuits that include basal-ganglia pathways.",
            "Toxoplasmosis can produce brain lesions that often involve the basal ganglia in people with weakened immunity.",
            "Huntington disease is inherited and progressively affects the striatum, part of the basal ganglia.",
        ],
    },
    {
        "id": "thalamus",
        "name": "Thalamus",
        "position": (11, -17, 9),
        "description": "Relays and coordinates sensory, motor, and alertness-related signals.",
        "conditions": [
            "A thalamic stroke or tumor can disrupt sensation, movement, or alertness.",
            "Some infections, including Japanese encephalitis, may produce thalamic brain lesions.",
            "Fatal familial insomnia is a rare inherited prion disease associated with thalamic degeneration.",
        ],
    },
    {
        "id": "hypothalamus",
        "name": "Hypothalamus",
        "position": (8, -4, -8),
        "description": "Regulates temperature, hunger, sleep, hormones, and autonomic functions.",
        "conditions": [
            "Craniopharyngiomas and other nearby tumors can disturb hypothalamic function.",
            "Inflammation or infection can affect the hypothalamus as part of a broader illness.",
            "Prader-Willi syndrome is genetic and can disrupt hypothalamic appetite and hormone regulation.",
        ],
    },
    {
        "id": "cerebellum",
        "name": "Cerebellum",
        "position": (0, -52, -31),
        "description": "Coordinates balance, timing, posture, and precise movement; it also contributes to learning.",
        "conditions": [
            "Cerebellar stroke or tumors, including medulloblastoma, can affect balance and coordination.",
            "Post-infectious cerebellitis can occur after an infection.",
            "Inherited spinocerebellar ataxias can progressively affect coordination.",
        ],
    },
    {
        "id": "pons",
        "name": "Pons",
        "position": (0, -32, -30),
        "description": "A brainstem bridge involved in sleep, breathing, facial sensation, and communication with the cerebellum.",
        "conditions": [
            "A pontine stroke can affect movement, sensation, or eye movements.",
            "Diffuse midline glioma is a serious tumor that can arise in the pons, particularly in children.",
            "Some brainstem encephalitides can involve the pons.",
        ],
    },
    {
        "id": "medulla",
        "name": "Medulla oblongata",
        "position": (0, -39, -47),
        "description": "The lowest brainstem region; it helps regulate breathing, heart rate, swallowing, and protective reflexes.",
        "conditions": [
            "A medullary stroke can cause swallowing, balance, or other brainstem symptoms.",
            "Brainstem inflammation or encephalitis can involve the medulla.",
            "Motor-neuron disorders such as ALS may affect motor pathways in the medulla as well as elsewhere.",
        ],
    },
    {
        "id": "spinal-cord",
        "name": "Spinal cord",
        "position": (0, -40, -74),
        "description": "Carries signals between the brain and body and coordinates spinal reflexes.",
        "conditions": [
            "Transverse myelitis and spinal-cord infections cause inflammation; multiple sclerosis can also affect the cord.",
            "Spinal-cord tumors and traumatic injury can interrupt nerve pathways.",
            "Inherited disorders such as hereditary spastic paraplegia or spinal muscular atrophy affect spinal motor pathways or motor neurons.",
        ],
    },
]

brain_view = components.declare_component(
    "neuro_explorer_brain",
    path=str(Path(__file__).parent / "brain_view"),
)


@st.cache_resource(show_spinner="Loading the folded cortical surface...")
def load_brain_surfaces():
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    hemispheres = []
    for mesh_path, sulc_path in (
        (fsaverage.pial_left, fsaverage.sulc_left),
        (fsaverage.pial_right, fsaverage.sulc_right),
    ):
        mesh = surface.load_surf_mesh(mesh_path)
        hemispheres.append(
            (
                np.asarray(mesh.coordinates),
                np.asarray(mesh.faces),
                np.asarray(surface.load_surf_data(sulc_path)).ravel(),
            )
        )
    return tuple(hemispheres)


def _add_ellipsoid(
    fig: go.Figure,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    name: str,
    color: str,
    opacity: float = 0.9,
) -> None:
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 16)
    x = center[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radii[2] * np.outer(np.ones_like(u), np.cos(v))
    rows, columns = x.shape
    i, j, k = [], [], []
    for row in range(rows - 1):
        for column in range(columns - 1):
            first = row * columns + column
            second = first + columns
            i.extend((first, second))
            j.extend((second, second + 1))
            k.extend((first + 1, first + 1))
    fig.add_trace(
        go.Mesh3d(
            x=x.ravel(),
            y=y.ravel(),
            z=z.ravel(),
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=opacity,
            flatshading=False,
            lighting={"ambient": 0.58, "diffuse": 0.72, "specular": 0.2},
            hovertemplate=f"{name} · stylized anatomical model<extra></extra>",
            name=name,
            showlegend=False,
        )
    )


def _add_spinal_cord(fig: go.Figure) -> None:
    angles = np.linspace(0, 2 * np.pi, 24)
    levels = np.linspace(-90, -51, 24)
    x, y, z = [], [], []
    for level in levels:
        bend = -40 - max(0, -level - 68) * 0.12
        x.extend(3.2 * np.cos(angles))
        y.extend(bend + 3.2 * np.sin(angles))
        z.extend([level] * len(angles))
    i, j, k = [], [], []
    for level in range(len(levels) - 1):
        for angle in range(len(angles)):
            nxt = (angle + 1) % len(angles)
            a = level * len(angles) + angle
            b = level * len(angles) + nxt
            c = (level + 1) * len(angles) + angle
            d = (level + 1) * len(angles) + nxt
            i.extend((a, b))
            j.extend((c, c))
            k.extend((b, d))
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            color="#d58c9b",
            opacity=0.9,
            lighting={"ambient": 0.6, "diffuse": 0.7},
            hovertemplate="Spinal cord · stylized anatomical model<extra></extra>",
            name="Spinal cord",
            showlegend=False,
        )
    )


def _add_full_anatomy(fig: go.Figure) -> None:
    for center, radii, name, color in (
        ((0, -52, -31), (34, 31, 16), "Cerebellum", "#b993bd"),
        ((0, -31, -30), (10, 12, 10), "Pons", "#d88c9b"),
        ((0, -39, -47), (7.5, 9, 12), "Medulla oblongata", "#d58c9b"),
    ):
        _add_ellipsoid(fig, center, radii, name, color)

    for name, center, radii, color in (
        ("Basal ganglia", (18, 3, 10), (8, 12, 10), "#d8a85f"),
        ("Thalamus", (11, -17, 9), (7, 10, 9), "#71b9c9"),
        ("Hypothalamus", (8, -4, -8), (5, 7, 5), "#c98470"),
        ("Hippocampus", (25, -22, -12), (5, 15, 5), "#9c90d0"),
        ("Amygdala", (24, -6, -12), (5, 7, 5), "#db8292"),
    ):
        _add_ellipsoid(fig, center, radii, name, color)
        _add_ellipsoid(
            fig,
            (-center[0], center[1], center[2]),
            radii,
            name,
            color,
        )

    _add_spinal_cord(fig)


def _add_region_markers(fig: go.Figure, selected_region: str | None) -> None:
    positions = np.asarray([region["position"] for region in REGIONS])
    label_positions = np.asarray(
        [REGION_LABEL_POSITIONS[region["id"]] for region in REGIONS]
    )
    ids = [region["id"] for region in REGIONS]
    names = [region["name"] for region in REGIONS]
    colors = [
        "#ffd166" if region_id == selected_region else "#71d7cf"
        for region_id in ids
    ]
    sizes = [15 if region_id == selected_region else 10 for region_id in ids]
    connector_x: list[float | None] = []
    connector_y: list[float | None] = []
    connector_z: list[float | None] = []
    for marker, label in zip(positions, label_positions):
        middle = marker * 0.65 + label * 0.35
        connector_x.extend((marker[0], middle[0], label[0], None))
        connector_y.extend((marker[1], middle[1], label[1], None))
        connector_z.extend((marker[2], middle[2], label[2], None))
    fig.add_trace(
        go.Scatter3d(
            x=connector_x,
            y=connector_y,
            z=connector_z,
            mode="lines",
            line={"color": "#91a7b7", "width": 1},
            hoverinfo="skip",
            name="Region label guides",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            customdata=ids,
            text=names,
            marker={
                "size": sizes,
                "color": colors,
                "opacity": 1,
                "line": {"color": "#ffffff", "width": 1},
            },
            hovertemplate="<b>%{text}</b><br>Click or double-click to inspect<extra></extra>",
            meta={"role": "brain-region-markers"},
            name="Brain regions",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=label_positions[:, 0],
            y=label_positions[:, 1],
            z=label_positions[:, 2],
            mode="text",
            text=names,
            textfont={"color": "#f0f3f6", "size": 10},
            hoverinfo="skip",
            name="Brain region labels",
            showlegend=False,
        )
    )


def simulate_hodgkin_huxley(stimulus_current: float) -> dict[str, object]:
    time_ms = np.arange(2001, dtype=float) * 0.01
    voltage_mv = np.empty_like(time_ms)
    sodium_current = np.empty_like(time_ms)
    potassium_current = np.empty_like(time_ms)

    def rates(voltage: float) -> tuple[float, float, float, float, float, float]:
        def x_over_one_minus_exp(x: float, scale: float) -> float:
            scaled = x / scale
            if abs(scaled) < 1e-6:
                return scale * (1 + scaled / 2 + scaled * scaled / 12)
            return x / -np.expm1(-scaled)

        alpha_m = 0.1 * x_over_one_minus_exp(voltage + 40, 10)
        beta_m = 4 * np.exp(-(voltage + 65) / 18)
        alpha_h = 0.07 * np.exp(-(voltage + 65) / 20)
        beta_h = 1 / (1 + np.exp(-(voltage + 35) / 10))
        alpha_n = 0.01 * x_over_one_minus_exp(voltage + 55, 10)
        beta_n = 0.125 * np.exp(-(voltage + 65) / 80)
        return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n

    voltage = -65.0
    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = rates(voltage)
    m = alpha_m / (alpha_m + beta_m)
    h = alpha_h / (alpha_h + beta_h)
    n = alpha_n / (alpha_n + beta_n)
    dt = float(time_ms[1] - time_ms[0])

    for index, time in enumerate(time_ms):
        voltage_mv[index] = voltage
        sodium_current[index] = 120 * m**3 * h * (voltage - 50)
        potassium_current[index] = 36 * n**4 * (voltage + 77)
        if index == len(time_ms) - 1:
            break

        injected_current = stimulus_current if 1.0 <= time < 2.0 else 0.0
        leak_current = 0.3 * (voltage + 54.4)
        voltage += dt * (
            injected_current
            - sodium_current[index]
            - potassium_current[index]
            - leak_current
        )
        alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = rates(voltage)
        m += dt * (alpha_m * (1 - m) - beta_m * m)
        h += dt * (alpha_h * (1 - h) - beta_h * h)
        n += dt * (alpha_n * (1 - n) - beta_n * n)

    crossings = np.flatnonzero((voltage_mv[:-1] <= 0) & (voltage_mv[1:] > 0))
    spike_latency_ms = (
        float(time_ms[crossings[0]]) if crossings.size else None
    )
    return {
        "time_ms": time_ms,
        "voltage_mv": voltage_mv,
        "sodium_current": sodium_current,
        "potassium_current": potassium_current,
        "stimulus_current": stimulus_current,
        "spike_latency_ms": spike_latency_ms,
        "peak_voltage_mv": float(np.max(voltage_mv)),
        "spiked": spike_latency_ms is not None,
    }


def _add_neural_network(
    fig: go.Figure,
    simulation: dict[str, object],
    stimulus_rate_hz: float,
    excitatory_input_mv: float,
) -> None:
    line_x: list[float | None] = []
    line_y: list[float | None] = []
    line_z: list[float | None] = []
    branch_x: list[float | None] = []
    branch_y: list[float | None] = []
    branch_z: list[float | None] = []
    cell_x: list[float] = []
    cell_y: list[float] = []
    cell_z: list[float] = []
    signal_paths = []
    node_x: list[float] = []
    node_y: list[float] = []
    node_z: list[float] = []
    route_offsets = np.linspace(-25, 25, 7)
    progress_steps = np.linspace(0, 1, 5)
    conduction_velocity_m_s = 10.0
    synaptic_delay_ms = 0.8
    integration_delay_ms = 2.5
    relay_threshold_mv = 10.0
    relay_enabled = excitatory_input_mv >= relay_threshold_mv
    spike_latency_ms = simulation["spike_latency_ms"]

    for route_index, offset in enumerate(route_offsets):
        neurons = []
        for neuron_index, progress in enumerate(progress_steps):
            point = np.array(
                [
                    offset + 13 * np.sin(progress * np.pi * 1.4 + route_index * 0.38),
                    36 - 82 * progress,
                    28 + 9 * np.sin(progress * np.pi + route_index * 0.3),
                ]
            )
            neurons.append(point)
            cell_x.append(point[0])
            cell_y.append(point[1])
            cell_z.append(point[2])

            for branch in range(4):
                angle = branch * np.pi / 2 + route_index * 0.45
                endpoint = point + np.array(
                    [5 * np.cos(angle), 4 * np.sin(angle), 5 * np.sin(angle + 0.7)]
                )
                branch_x.extend((point[0], endpoint[0], None))
                branch_y.extend((point[1], endpoint[1], None))
                branch_z.extend((point[2], endpoint[2], None))

        route_spike_time = (
            float(spike_latency_ms) + route_index * 0.6
            if bool(simulation["spiked"])
            else None
        )
        for edge_index, (start, end) in enumerate(zip(neurons, neurons[1:])):
            midpoint = (start + end) / 2 + np.array([0, -2, 4])
            curve = np.linspace(0.08, 0.92, 9)

            def point_on_axon(progress: float) -> np.ndarray:
                inverse = 1 - progress
                return (
                    inverse**2 * start
                    + 2 * inverse * progress * midpoint
                    + progress**2 * end
                )

            nodes = np.asarray([point_on_axon(float(p)) for p in curve])
            node_x.extend(nodes[:, 0])
            node_y.extend(nodes[:, 1])
            node_z.extend(nodes[:, 2])
            for node_index in range(len(curve) - 1):
                beginning = point_on_axon(float(curve[node_index] + 0.025))
                finish = point_on_axon(float(curve[node_index + 1] - 0.025))
                line_x.extend((beginning[0], finish[0], None))
                line_y.extend((beginning[1], finish[1], None))
                line_z.extend((beginning[2], finish[2], None))

            length_m = 0.035 + route_index * 0.001 + edge_index * 0.002
            conduction_ms = length_m / conduction_velocity_m_s * 1000
            can_relay = (
                bool(simulation["spiked"])
                and (edge_index == 0 or relay_enabled)
            )
            path_start_ms = route_spike_time if can_relay else None

            signal_paths.append(
                {
                    "nodes": nodes.tolist(),
                    "terminal": point_on_axon(0.95).tolist(),
                    "target": end.tolist(),
                    "startTimeMs": path_start_ms,
                    "conductionMs": conduction_ms,
                    "synapticDelayMs": synaptic_delay_ms,
                    "lengthMm": length_m * 1000,
                }
            )
            if path_start_ms is not None and relay_enabled:
                route_spike_time = (
                    path_start_ms
                    + conduction_ms
                    + synaptic_delay_ms
                    + integration_delay_ms
                )
            else:
                route_spike_time = None

    fig.add_trace(
        go.Scatter3d(
            x=branch_x,
            y=branch_y,
            z=branch_z,
            mode="lines",
            line={"color": "#71d7cf", "width": 2},
            hoverinfo="skip",
            name="Dendrites",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            line={"color": "#c5a1f2", "width": 4},
            hoverinfo="skip",
            name="Myelinated internodes",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            marker={
                "size": 3,
                "color": "#f0d7a2",
                "line": {"color": "#fff3c4", "width": 1},
            },
            hovertemplate="Node of Ranvier · action potentials are regenerated here<extra></extra>",
            name="Nodes of Ranvier",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=cell_x,
            y=cell_y,
            z=cell_z,
            mode="markers",
            marker={
                "size": 5,
                "color": "#8bded3",
                "line": {"color": "#e9fff8", "width": 1},
            },
            hovertemplate="Neuron cell body (soma)<extra></extra>",
            name="Neuron cell bodies",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="markers",
            marker={
                "size": [],
                "color": [],
                "line": {"color": "#fff3c4", "width": 1},
            },
            hovertemplate="Action potential · %{customdata:.0f} mV<extra></extra>",
            customdata=[],
            meta={
                "role": "action-potential-nodes",
                "paths": signal_paths,
                "spiked": simulation["spiked"],
            },
            name="Action potentials",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="markers",
            marker={
                "size": 5,
                "color": "#ffcb69",
                "line": {"color": "#fff3c4", "width": 1},
            },
            hovertemplate="Glutamate molecules crossing the synaptic cleft<extra></extra>",
            meta={"role": "transmitter-particles"},
            name="Glutamate release",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="markers",
            marker={
                "size": [],
                "color": "#71d7cf",
                "opacity": 0.85,
                "line": {"color": "#d5fff8", "width": 1},
            },
            hovertemplate="Graded excitatory postsynaptic potential (EPSP)<extra></extra>",
            meta={"role": "postsynaptic-epsp"},
            name="Postsynaptic potentials",
            showlegend=False,
        )
    )


def _show_membrane_response(simulation: dict[str, object]) -> None:
    time_ms = np.asarray(simulation["time_ms"])
    voltage_mv = np.asarray(simulation["voltage_mv"])
    response = go.Figure()
    response.add_trace(
        go.Scatter(
            x=time_ms,
            y=voltage_mv,
            mode="lines",
            line={"color": "#71d7cf", "width": 2.5},
            name="Membrane voltage",
            hovertemplate="Time: %{x:.2f} ms<br>Voltage: %{y:.1f} mV<extra></extra>",
        )
    )
    response.add_trace(
        go.Scatter(
            x=time_ms,
            y=np.asarray(simulation["sodium_current"]),
            mode="lines",
            line={"color": "#d888b7", "width": 1.5},
            name="Na⁺ current",
            yaxis="y2",
            hovertemplate="Time: %{x:.2f} ms<br>Na⁺ current: %{y:.1f} µA/cm²<extra></extra>",
        )
    )
    response.add_trace(
        go.Scatter(
            x=time_ms,
            y=np.asarray(simulation["potassium_current"]),
            mode="lines",
            line={"color": "#8fa8f8", "width": 1.5},
            name="K⁺ current",
            yaxis="y2",
            hovertemplate="Time: %{x:.2f} ms<br>K⁺ current: %{y:.1f} µA/cm²<extra></extra>",
        )
    )
    response.add_hline(
        y=-55,
        line={"color": "#ffd166", "dash": "dash"},
        annotation_text="Approx. spike threshold",
        annotation_position="top left",
        annotation_font_color="#ffd166",
    )
    response.add_hline(
        y=-65,
        line={"color": "#8793a3", "dash": "dot"},
        annotation_text="Resting potential",
        annotation_position="bottom left",
        annotation_font_color="#9ca8b8",
    )
    response.add_vrect(
        x0=1,
        x1=2,
        fillcolor="#ffd166",
        opacity=0.12,
        line_width=0,
        annotation_text="current pulse",
        annotation_position="top left",
        annotation_font_color="#ffd166",
    )
    response.update_layout(
        title=(
            f"Current clamp: {simulation['stimulus_current']:g} µA/cm² for 1 ms · "
            "Hodgkin–Huxley membrane response"
        ),
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font={"color": TEXT},
        margin={"l": 10, "r": 55, "t": 55, "b": 10},
        height=300,
        xaxis={
            "title": "Time (ms)",
            "gridcolor": "#293341",
            "zerolinecolor": "#293341",
        },
        yaxis={
            "title": "Membrane potential (mV)",
            "range": [-85, max(45, float(simulation["peak_voltage_mv"]) + 8)],
            "gridcolor": "#293341",
            "zerolinecolor": "#293341",
        },
        yaxis2={
            "title": "Ionic current (µA/cm²)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "zeroline": False,
        },
        legend={"orientation": "h", "y": -0.22, "x": 0.1},
        showlegend=True,
    )
    st.plotly_chart(
        response,
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )


def build_figure(
    surfaces,
    mode: str,
    selected_region: str | None = None,
    simulation: dict[str, object] | None = None,
    stimulus_rate_hz: float = 4,
    excitatory_input_mv: float = 12,
) -> go.Figure:
    all_coords = np.concatenate([hemisphere[0] for hemisphere in surfaces])
    lower = np.percentile(all_coords, 1, axis=0)
    upper = np.percentile(all_coords, 99, axis=0)
    sulcal_values = np.concatenate([hemisphere[2] for hemisphere in surfaces])
    color_min, color_max = np.percentile(sulcal_values, [2, 98])
    fig = go.Figure()

    for coords, faces, sulc in surfaces:
        fig.add_trace(
            go.Mesh3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=sulc,
                intensitymode="vertex",
                colorscale=[[0, "#343a48"], [0.48, "#9299a7"], [1, "#e4c4a4"]],
                cmin=color_min,
                cmax=color_max,
                showscale=False,
                opacity=1 if mode == "Surface" else (0.36 if mode == "Full anatomy" else 0.13),
                flatshading=False,
                lighting={
                    "ambient": 0.48,
                    "diffuse": 0.78,
                    "specular": 0.18,
                    "roughness": 0.82,
                    "fresnel": 0.08,
                },
                lightposition={"x": 120, "y": 180, "z": 220},
                hoverinfo="skip",
                name="Cortical surface",
                showlegend=False,
            )
        )

    if mode != "Surface":
        _add_full_anatomy(fig)
    if mode == "Full anatomy":
        _add_region_markers(fig, selected_region)
    if mode == "Neural signals":
        if simulation is None:
            simulation = simulate_hodgkin_huxley(12)
        _add_neural_network(
            fig,
            simulation,
            stimulus_rate_hz,
            excitatory_input_mv,
        )

    all_x = [lower[0] - 8, upper[0] + 8]
    all_y = [lower[1] - 12, upper[1] + 12]
    all_z = [-96, upper[2] + 8]
    fig.update_layout(
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
        clickmode="event",
        scene={
            "bgcolor": BACKGROUND,
            "xaxis": {"visible": False, "range": all_x},
            "yaxis": {"visible": False, "range": all_y},
            "zaxis": {"visible": False, "range": all_z},
            "aspectmode": "data",
            "camera": {
                "eye": {"x": 1.7, "y": 1.45, "z": 1.05},
                "up": {"x": 0, "y": 0, "z": 1},
            },
        },
        uirevision=mode,
    )
    return fig


def _show_region_details(region: dict) -> None:
    st.markdown(f"### {region['name']}")
    st.write(region["description"])
    st.markdown("**Examples of associated conditions**")
    for condition in region["conditions"]:
        st.markdown(f"- {condition}")
    st.caption(
        "Educational examples only, not a diagnosis or a complete list. "
        "Many neurological conditions affect multiple regions; a dot does not "
        "mean a condition is limited to that structure."
    )


def main() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {BACKGROUND}; color: {TEXT}; }}
        [data-testid="stSidebar"] {{ background: {PANEL}; }}
        .block-container {{ padding-top: 2rem; }}
        .eyebrow {{ color: {ACCENT}; font-size: .78rem; font-weight: 700;
                    letter-spacing: .16em; text-transform: uppercase; }}
        .subtitle {{ color: {MUTED}; font-size: 1.05rem; }}
        .caption-card {{ background: {PANEL}; border: 1px solid #293341;
                         border-radius: 12px; padding: 1rem 1.2rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">Neuroscience · 3D explorer</div>', unsafe_allow_html=True)
    st.title("Inside the human brain")
    st.markdown(
        '<p class="subtitle">Explore the brain from its folded surface to its '
        'deep structures, then watch signals travel through a neuron network.</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## View")
        mode = st.radio(
            "Choose a brain view",
            ["Surface", "Full anatomy", "Neural signals"],
            label_visibility="collapsed",
        )
        st.divider()
        if mode == "Surface":
            st.markdown("### Cortical surface")
            st.write(
                "Rotate the cortical surface to explore its gyri (ridges) and "
                "sulci (grooves)."
            )
        elif mode == "Full anatomy":
            st.markdown("### Brain regions")
            st.write(
                "Click or double-click a turquoise dot on the model to see the region's "
                "function and example associated conditions."
            )
            chosen_name = st.selectbox(
                "Or select a region",
                ["Choose a region…", *(region["name"] for region in REGIONS)],
                key="region_picker",
            )
            if chosen_name != "Choose a region…":
                selected_region = next(
                    region for region in REGIONS if region["name"] == chosen_name
                )
                st.session_state["selected_brain_region"] = selected_region["id"]
            st.caption(
                "Includes the cerebellum, deep nuclei, brainstem, medulla oblongata, "
                "and spinal cord."
            )
        else:
            st.markdown("### Physiological-style stimulation")
            st.write(
                "Stimulate a Hodgkin–Huxley neuron and follow action potentials "
                "along myelinated axons, through glutamate synapses, and into "
                "postsynaptic neurons."
            )
            stimulus_current = st.slider(
                "Injected current (µA/cm²)",
                min_value=0.0,
                max_value=20.0,
                value=12.0,
                step=0.5,
                help="A 1 ms current pulse is applied to the modeled source neuron.",
            )
            stimulus_rate_hz = st.slider(
                "Stimulus rate (Hz)",
                min_value=1,
                max_value=8,
                value=4,
                help="Stimulus pulses are spaced to allow membrane recovery between spikes.",
            )
            excitatory_input_mv = st.slider(
                "Summed excitatory input (mV)",
                min_value=0.0,
                max_value=20.0,
                value=12.0,
                step=1.0,
                help=(
                    "Simplified compound EPSP at the next neuron. The model uses "
                    "10 mV as an illustrative relay threshold."
                ),
            )
            playback_speed = st.slider(
                "Playback speed",
                min_value=0.4,
                max_value=2.0,
                value=1.0,
                step=0.1,
                format="%.1f×",
            )
            st.caption(
                "Model: Hodgkin–Huxley Na⁺/K⁺/leak currents, saltatory conduction "
                "(10 m/s), glutamate release, and a thresholded compound EPSP. "
                "Uses classic squid-axon parameters to illustrate general "
                "membrane physiology; signals are slowed 20× for visibility."
            )

    surfaces = load_brain_surfaces()
    selected_region_id = st.session_state.get("selected_brain_region")
    selected_region = next(
        (region for region in REGIONS if region["id"] == selected_region_id),
        None,
    )
    simulation = (
        simulate_hodgkin_huxley(stimulus_current)
        if mode == "Neural signals"
        else None
    )
    fig = build_figure(
        surfaces,
        mode,
        selected_region["id"] if selected_region else None,
        simulation,
        stimulus_rate_hz if mode == "Neural signals" else 4,
        excitatory_input_mv if mode == "Neural signals" else 12,
    )
    is_spiking = bool(simulation and simulation["spiked"])
    result = brain_view(
        figure=json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
        animate=mode == "Neural signals",
        spiking=is_spiking,
        voltageTime=simulation["time_ms"].tolist() if simulation else [],
        voltage=simulation["voltage_mv"].tolist() if simulation else [],
        apLatency=simulation["spike_latency_ms"] if simulation else None,
        firingRate=stimulus_rate_hz if mode == "Neural signals" else 4,
        relay=excitatory_input_mv >= 10 if mode == "Neural signals" else False,
        inputAmplitude=excitatory_input_mv if mode == "Neural signals" else 0,
        speed=playback_speed if mode == "Neural signals" else 1.0,
        selectedRegion=selected_region_id,
        key="brain_model",
        default=None,
        height=740,
    )
    if mode == "Full anatomy" and isinstance(result, dict):
        region_id = result.get("region")
        activation = result.get("activatedAt")
        if (
            activation != st.session_state.get("last_region_activation")
            and region_id in {region["id"] for region in REGIONS}
        ):
            st.session_state["last_region_activation"] = activation
            if st.session_state.get("selected_brain_region") != region_id:
                st.session_state["selected_brain_region"] = region_id
                st.rerun()
            st.session_state["selected_brain_region"] = region_id
            selected_region = next(
                region for region in REGIONS if region["id"] == region_id
            )
    if mode == "Full anatomy":
        if selected_region:
            _show_region_details(selected_region)
        else:
            st.info("Double-click one of the labeled dots to inspect a brain region.")

    if mode == "Surface":
        st.markdown(
            '<div class="caption-card">Surface geometry: fsaverage5 cortical pial meshes. '
            'Color shading reflects sulcal depth. Drag to rotate; scroll to zoom.</div>',
            unsafe_allow_html=True,
        )
    elif mode == "Full anatomy":
        st.markdown(
            '<div class="caption-card">Cortical geometry uses fsaverage5 surface meshes. '
            'Deep structures, cerebellum, brainstem, and spinal cord are simplified '
            'educational shapes, not a clinical or patient-specific scan.</div>',
            unsafe_allow_html=True,
        )
    else:
        _show_membrane_response(simulation)
        if not is_spiking:
            st.warning(
                "This current pulse is subthreshold in the modeled neuron, so it "
                "produces no action potential to propagate. Increase the injected "
                "current to trigger a spike."
            )
        st.markdown(
            '<div class="caption-card">The membrane waveform uses the classic '
            'Hodgkin–Huxley conductance model (voltage-gated sodium, potassium, and '
            'leak currents), whose original parameters describe the squid giant '
            'axon—not human neurons. The displayed brain geometry is schematic; '
            'axon lengths, conduction, synaptic integration, and the compound EPSP '
            'are simplified teaching assumptions, not a human-specific or clinical '
            'simulation.</div>',
            unsafe_allow_html=True,
        )


def render_footer():
    st.sidebar.divider()
    st.sidebar.caption(
       "Educational use only. Not medical advice, and not a diagnosis tool."
    )
    st.sidebar.markdown(
        """
**Sources**
- Cortical surface: fsaverage template (FreeSurfer), loaded with nilearn
- Neuron model: Hodgkin & Huxley (1952), *J Physiol* 117:500-544
- Built with Streamlit, nilearn and Plotly

**Made by Teagan🤘🏼**

Feedback: [LinkedIn](https://www.linkedin.com/in/teagan-sharma-a8ba77399) · [GitHub Issues](https://github.com/teagansstudy-ui/neuro-python/issues)
    )


if __name__ == "__main__":
    main()
    render_footer()
    
