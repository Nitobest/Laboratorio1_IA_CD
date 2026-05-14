# data/raw/

Carpeta para los datos originales sin modificar.

## Cómo obtener el dataset crudo

El Dry Bean Dataset se descarga programáticamente desde UCI usando `ucimlrepo`:

```python
from ucimlrepo import fetch_ucirepo
dry_bean = fetch_ucirepo(id=602)
X = dry_bean.data.features
y = dry_bean.data.targets
```

Fuente oficial: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

- 13.611 instancias originales (incluye duplicados)
- 16 variables predictoras + 1 variable objetivo (`Class`)
- 7 clases de frijol: DERMASON, SIRA, SEKER, HOROZ, CALI, BARBUNYA, BOMBAY

El archivo limpio (sin duplicados, 13.543 filas) se guarda en `../processed/dry_bean_clean.csv` al ejecutar el notebook de preparación.
