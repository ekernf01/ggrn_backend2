# Differentiable Causal Discovery with Factor Graphs

This code is an interface to software for ["Large-Scale Differentiable Causal Discovery of Factor Graphs"](https://arxiv.org/abs/2206.07824), lightly modified from the [original repo](https://github.com/Genentech/dcdfg). That repository was originally forked from [DCDI](https://github.com/slachapelle/dcdi). Please refer to the NOTICE file for licensing information.

## Usage

    import dcdfg_wrapper
    factor_graph_model = DCDFGWrapper()
    factor_graph_model.train(my_anndata)
    predictions_anndata = factor_graph_model.predict([('POU5F1', 5), ('NANOG', 0)])
    
### Input data

Input data are expected to be AnnData objects with several specific metadata fields, such as "perturbation". For context, this wrapper over DCD-FG is part of a benchmarking study that includes a collection of RNA-seq data all formatted in a standard way. See our [benchmarking repo](https://github.com/ekernf01/perturbation_benchmarking) for more information. To load a working example, use this code from our benchmarking infrastructure. 
    
```python
import load_perturbations
load_perturbations.set_data_path(
    '../perturbation_data/perturbations' # Change this to wherever you placed our collection of perturbation data.
)
my_anndata = load_perturbations.load_perturbation("nakatake")
```

### Dependencies

- For stand-alone use of this package, there is a `requirements.txt` so you can use pip to install it and the deps should be taken care of.
- This is part of a benchmarking study that includes a conda environment with exact deps pinned. See our benchmarking repo for more information. 
