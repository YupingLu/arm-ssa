# arm-ssa
Singular spectrum analysis to detect outliers in ARM data. This is a univariate analysis method. You need to pick the variable first.

1. Calculate outliers and visualize the results
```
  arm-ssa.py: python arm-ssa.py
  # Figures are stored in ssa-figures and ssa-figures-outliers
```
2. Query DQR database
```
  db.py: python db.py
  # Results are stored in db.records
```

3. Visualize outliers in Plotly
```
  arm-ssa-plotly.py: python arm-ssa-plotly.py
  # Figures are stored in plotly
```
