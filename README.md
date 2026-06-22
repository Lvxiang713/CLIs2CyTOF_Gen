# Single-cell generative modeling from clinical laboratory indicators

This repository contains code for conditional generation of CyTOF-based single-cell protein profiles from clinical laboratory indicators. The implemented methods include Flow Matching, DDPM, VAE, and GAN.

## Repository structure

```text
analysis/      Analysis pipeline and notebook utilities
configs/       Unified configuration dataclass
data/          Dataset loaders for EHR and single-cell protein data
evaluation/    Generation evaluation utilities
models/        Model backbones, conditioning modules, and method wrappers
sampling/      Chunked sampling utilities
training/      Training utilities and trainers
run.py         Unified training and sampling entry point
run_analysis.py Unified downstream analysis entry point
requirements.txt
```

## Environment setup

```bash
conda create -n singlecell-gen python=3.9 -y
conda activate singlecell-gen
pip install -r requirements.txt
```


## Basic usage

Run training and generation through the unified entry point:

```bash
python run.py \
  --method flowmatching \
  --ehr_csv your_ehr_csv_datapath \
  --sc_csv your_single_cell_csv_datapath \
  --label_xlsx your_label_xlsx_datapath
```

The supported method names are:

```text
flowmatching, ddpm, vae, gan
```

You can also use the method-specific wrapper scripts:

```bash
python run_flow_matching.py
python run_ddpm.py
python run_vae.py
python run_gan.py
```
The trained checkpoints in this work in available at: https://zenodo.org/records/20794836

## Downstream analysis

After generation, run the analysis pipeline with:

```bash
python run_analysis.py \
  --method flowmatching \
  --ehr_csv your_ehr_csv_datapath \
  --sc_csv your_single_cell_csv_datapath \
  --label_xlsx your_label_xlsx_datapath
```

For notebook-based analysis, open:

```text
analysis/Analysis_demo.ipynb
```

and update the placeholder paths in the first configuration cell.
