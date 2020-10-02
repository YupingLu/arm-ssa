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
Reference Paper
---------------
Y. Lu, J. Kumar, N. Collier, B. Krishna and M. A. Langston, "Detecting Outliers in Streaming Time Series Data from ARM Distributed Sensors," 2018 IEEE International Conference on Data Mining Workshops (ICDMW), Singapore, Singapore, 2018, pp. 779-786, doi: 10.1109/ICDMW.2018.00117.
