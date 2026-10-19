# Leaderboard

This directory contains the MIMO-OFDM channel-estimation leaderboard. Sphinx
builds copy its static assets into the generated documentation tree through
`doc/conf.py`; there is no separately maintained documentation copy.

Generate the dashboard data from the supplied canonical outputs, then serve it
from the repository root:

```bash
python leaderboard/mimo_ofdm_chest/generate_data.py
python3 -m http.server 8000 --directory leaderboard/mimo_ofdm_chest
```

Run the generator again after canonical benchmark results change:

```bash
python leaderboard/mimo_ofdm_chest/generate_data.py
```

To visualize a separate result tree, pass its root explicitly:

```bash
python leaderboard/mimo_ofdm_chest/generate_data.py --output-root generated-outputs
```
