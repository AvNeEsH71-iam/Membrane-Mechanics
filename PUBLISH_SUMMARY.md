# GitHub Publish Package

This folder is a clean, upload-ready bundle of the required project files.

## Included

- Website app:
  - `index.html`
  - `css/`
  - `js/`
  - `pages/`
- Project metadata:
  - `README.md`
  - `LICENSE`
  - `.gitignore`
- Equatorial fluctuation spectroscopy code:
  - `fluctuation-analysis/app.py`
  - `fluctuation-analysis/requirements.txt`
  - `fluctuation-analysis/quasi_spherical/` (all required pipeline modules)
- Legacy/reference source code:
  - `fluctuation-analysis/*.cpp`
  - `fluctuation-analysis/*.cu`
  - `fluctuation-analysis/plotting.py`
  - `fluctuation-analysis/plotting_analytical.py`
  - `fluctuation-analysis/review_analytical.py`
  - `fluctuation-analysis/CMakeLists.txt`
- Documentation/support:
  - `fluctuation-analysis/paper_extract_focus.txt`
  - `docs/Fluctuation analysis.pdf`

## Not included (intentionally)

- Build/cache/output artifacts (`build/`, `output*/`, `__pycache__/`, `.venv/`, generated plots/gifs/notebooks)
- These are runtime/generated files and not required for source publication.

## Quick run

### Website

```bash
python -m http.server 8000
# open http://127.0.0.1:8000/index.html
```

### Python equatorial pipeline

```bash
cd fluctuation-analysis
python -m quasi_spherical.main --fit-lmin 3 --fit-lmax 10
```

### Streamlit interface

```bash
cd fluctuation-analysis
pip install -r requirements.txt
streamlit run app.py
```
