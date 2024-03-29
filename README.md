# Differentiable Causal Discovery with Factor Graphs

This code is an interface to software for ["Large-Scale Differentiable Causal Discovery of Factor Graphs"](https://arxiv.org/abs/2206.07824), lightly modified from the [original repo](https://github.com/Genentech/dcdfg). That repository was originally forked from [DCDI](https://github.com/slachapelle/dcdi). Please refer to the NOTICE file for licensing information.

## Usage

    import ggrn_backend2.api as dcdfg_wrapper 
    factor_graph_model = dcdfg_wrapper.DCDFGWrapper()
    factor_graph_model.train(my_anndata)
    predictions_anndata = factor_graph_model.predict([('POU5F1', 5), ('NANOG', 0)])

Note: this code will generate Wandb logging files as a side effect.

### Input data

Input data are expected to be AnnData objects with several specific metadata fields, such as "perturbation". For context, this wrapper over DCD-FG is part of a benchmarking study that includes a collection of RNA-seq data all formatted in a standard way. See our [benchmarking repo](https://github.com/ekernf01/perturbation_benchmarking) or [data collection](https://github.com/ekernf01/perturbation_data) for more information. To load a suitable example dataset, install our benchmarking infrastructure and use the following code. 
    
```python
import pereggrn_perturbations
pereggrn_perturbations.set_data_path(
    '../perturbation_data/perturbations' # Change this to wherever you placed our collection of perturbation data.
)
my_anndata = pereggrn_perturbations.load_perturbation("nakatake")
```

### Dependencies

- For stand-alone use of this package, there is a `requirements.txt` so you can use pip to install it and the deps should be taken care of.
- This is part of a benchmarking study that includes a conda environment with exact deps pinned. See our benchmarking repo for more information. 
