# Drifto

Drifto is an educational Streamlit dashboard for exploring model performance and feature drift on tabular classification data. It trains a Random Forest model, simulates shifted data, and compares the simulated sample with a held-out reference split.

> Drifto is a local prototype. It does not connect to a live production model or data stream, and simulated drift is not evidence that a deployed model has degraded.

## Current features

- Built-in scikit-learn datasets: Breast Cancer, Iris, and Wine.
- CSV upload with a selectable target column. Numeric feature columns are supported; categorical feature encoding is not implemented.
- Missing target rows are discarded. Missing numeric feature values are filled using medians calculated from the training split. Infinite values and unsupported feature types are rejected with an explanation.
- Random Forest classification with baseline accuracy, weighted precision, weighted recall, and weighted F1.
- Reproducible simulated data with a seed, shift strength, and Gentle, Standard, or Strong scenario preset.
- Feature drift scoring using absolute mean difference divided by the reference standard deviation. Features at or above the selected threshold are flagged.
- Dashboard views for overview, baseline metrics, prediction distribution, feature drift analysis, drift warnings, and project information.
- JSON run-summary and CSV row-result downloads from the Overview page.
- Plotly charts, with an Altair fallback if Plotly is unavailable.

## How the dashboard works

1. Choose a built-in dataset or upload a CSV and select its target column.
2. Drifto creates a stratified training/reference split and trains a Random Forest on the training rows.
3. It simulates changes to up to six numeric features in the reference rows. The random seed makes the simulation repeatable for the same settings.
4. It predicts the simulated rows and compares predictions against the retained reference labels. This is a controlled experiment, not independently observed production performance.
5. It calculates a standardized mean-shift score for each numeric feature and flags scores that meet the configured threshold.
6. Explore charts and tables, adjust settings, or download the run summary and row-level results.

The drift score is:

```text
abs(simulated feature mean - reference feature mean) / reference feature standard deviation
```

A zero-variance reference feature receives a score of zero. This score is a simple effect-size indicator. It does not test distribution shape, feature relationships, statistical significance, or concept drift.

## Project structure

```text
.
|-- app.py                  # Streamlit dashboard
|-- requirements.txt        # Python dependencies
|-- README.md
|-- .gitignore
|-- src/
|   |-- data.py             # Built-in datasets and data helpers
|   |-- train.py            # Command-line baseline training pipeline
|   |-- simulate.py         # Command-line production-data simulation
|   |-- monitor.py          # Command-line drift report generation
|   `-- utils.py            # Shared file and console helpers
|-- models/                 # Generated model artifacts (ignored by Git)
`-- outputs/                # Generated data and reports (ignored by Git)
```

## Run locally

### Requirements

- Python 3.11 or later
- Git, if cloning the repository

### Install

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Start the dashboard

From the repository root, run:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

### Run the file-based pipeline

The command-line pipeline is separate from the dashboard and writes generated artifacts into `models/` and `outputs/`. Run its steps in order:

```bash
python -m src.train
python -m src.simulate
python -m src.monitor
```

These generated files are intentionally ignored by Git. They can be recreated by running the pipeline.

## Hosting status

The app is structured to run as a Streamlit application and lists its dependencies in `requirements.txt`. To host it, publish the repository to a Git provider and configure the hosting service to install `requirements.txt` and start `app.py` with `streamlit run app.py`.

Before sharing a hosted instance, review the host's resource and upload limits and decide whether uploaded datasets may be processed there. Uploaded data and derived datasets may remain in Streamlit's in-process cache for the life of the app process. Drifto has no sign-in flow, role controls, persistent run history, live monitoring connection, scheduled runs, or production alerting. Downloaded reports are generated for the user to save.

## Data and interpretation limits

- CSV features must be numeric. Target labels may be numeric or categorical.
- The demo simulates drift in selected numeric features and retains reference labels for controlled metric comparisons.
- A change in feature distributions or predicted class proportions does not prove performance degradation.
- Real performance monitoring requires verified labels from the monitored population and an appropriate evaluation process.
- The mean-shift score can miss changes in distribution shape and other forms of data or concept drift.
- The app does not automatically retrain, alert, deploy, or roll back models.

## Checks

The repository does not currently include a committed automated test suite. Useful local checks include:

```bash
python -m compileall -q app.py src
streamlit run app.py
```

The dashboard has also been exercised with Streamlit's `AppTest` during development, including its pages, scenario controls, and download buttons.

## Roadmap

- Add a maintained automated test suite and continuous integration.
- Support categorical feature encoding with explicit preprocessing controls.
- Compare mean-shift scores with statistical drift tests.
- Add persistent run history and monitoring against an authorized live source.
- Add authentication, access controls, and operational alerting before production use.

## Contributing

Bug reports and improvements are welcome. Please describe the expected behavior, steps to reproduce issues, and the environment used. Add or update tests when changing data validation, training, or drift calculations.

## License

No license is currently provided. Add a `LICENSE` file before redistributing this project under an open-source license.
