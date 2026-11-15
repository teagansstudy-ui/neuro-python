# Neuro Explorer
**Live Demo:**https://neuro-explorer.streamlit.app/
NOTE- EDUCATIONAL USE ONLY. NOT MEDICAL ADVICE.
An interactive 3D brain explorer with three views:

- **Surface** — rotate and zoom a folded cortical surface with gyri and sulci.
- **Full anatomy** — inspect labeled brain regions, including deep structures, the cerebellum, medulla, and spinal cord. Click or double-click a region marker (or choose it from the list) for its role and examples of associated conditions.
- **Neural signals** — stimulate a Hodgkin–Huxley neuron with a current pulse, follow action potentials along myelinated axons, and observe glutamate release and a graded postsynaptic response. Adjust the stimulus rate and excitatory input to see whether downstream neurons reach the relay threshold.

Neural signal animation is slowed 20× for visibility at the default playback speed.

The cortical geometry and sulcal-depth shading use the fsaverage5 surface distributed by
Nilearn. The deep structures and spinal cord are simplified educational shapes. The neural
view uses a numerical Hodgkin–Huxley membrane model and a simplified synapse/network;
classic Hodgkin–Huxley parameters describe the squid giant axon, not human neurons. Brain
geometry and pathway lengths are schematic, so this is not a human-specific or clinical
simulation. Condition examples are non-exhaustive and do not imply that a condition is
limited to one region. On first launch, Nilearn downloads and caches the surface dataset;
the interactive viewer loads Plotly.js from its CDN.

## Run the app

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```



The first startup downloads and caches the Nilearn surface dataset, so it may take
longer than later visits. The browser also needs internet access to load Plotly.js.

The existing `brain_info.py` and `atlas_view.py` scripts remain available for downloading
the MNI152 volume template and inspecting its metadata.
