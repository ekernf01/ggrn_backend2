import os
import anndata
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
import torch
torch.set_default_tensor_type(torch.DoubleTensor)
# Eventually we may try to remove this dependency entirely
os.environ['WANDB_MODE'] = 'offline'
os.environ["WANDB_SILENT"] = "true"
import wandb
from ggrn_backend2.dcdfg.callback import (AugLagrangianCallback, ConditionalEarlyStopping,
                            CustomProgressBar)
from ggrn_backend2.dcdfg.linear_baseline.model import LinearGaussianModel
from ggrn_backend2.dcdfg.lowrank_linear_baseline.model import LinearModuleGaussianModel
from ggrn_backend2.dcdfg.lowrank_mlp.model import MLPModuleGaussianModel
from ggrn_backend2.dcdfg.perturbseq_data import PerturbSeqDataset
from sklearn.model_selection import KFold
    
class DCDFGWrapper:
    def __init__(self):
        self.model = None
        self.train_dataset = None

    def train(
        self,
        adata: anndata.AnnData, 
        train_batch_size: int = 4096,
        val_batch_size: int = 4096,
        num_train_epochs: int = 600,
        num_fine_epochs: int = 100,
        num_modules: int = 20,
        learning_rate: float = 1e-3,
        regularization_parameter: float = 0.1,
        constraint_mode: str = "spectral_radius",
        model_type: str = "linearlr",    
        do_use_polynomials: bool = False,
        logfile: str = "logs",
        verbose: bool = False,
        num_gpus = 1 if torch.cuda.is_available() else 0,
    ):
        if regularization_parameter is None:
            regularization_parameter = 0.1
        train_dataset = PerturbSeqDataset(adata)
        print("Train dataset size", train_dataset.dim, len(train_dataset))
        nb_nodes = train_dataset.dim
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        self.train_dataset = train_dataset
        
        if model_type == "linear":
            # create model
            self.model = LinearGaussianModel(
                nb_nodes,
                lr_init=learning_rate,
                reg_coeff=regularization_parameter,
                constraint_mode=constraint_mode,
                poly=do_use_polynomials,
            )
        elif model_type == "linearlr":
            self.model = LinearModuleGaussianModel(
                nb_nodes,
                num_modules,
                lr_init=learning_rate,
                reg_coeff=regularization_parameter,
                constraint_mode=constraint_mode,
            )            
        elif model_type == "mlplr":
            self.model = MLPModuleGaussianModel(
                nb_nodes,
                2,
                num_modules,
                16,
                lr_init=learning_rate,
                reg_coeff=regularization_parameter,
                constraint_mode=constraint_mode,
            )
        elif model_type == "already_trained":
            assert self.model is not None
        else:
            raise ValueError("model_type should be linear or linearlr or mlplr or already_trained.")

        # LOG CONFIG
        logger = WandbLogger(project=logfile, log_model=True)
        model_name = self.model.__class__.__name__
        if do_use_polynomials and model_name == "LinearGaussianModel":
            model_name += "_poly"
        logger.experiment.config.update(
            {"model_name": model_name, "module_name": self.model.module.__class__.__name__}
        )
        
        # Step 1: augmented lagrangian
        early_stop_1_callback = ConditionalEarlyStopping(
            monitor="Val/aug_lagrangian",
            min_delta=1e-4,
            patience=5,
            verbose=verbose,
            mode="min",
        )
        trainer = pl.Trainer(
            num_nodes=num_gpus,
            max_epochs=num_train_epochs,
            logger=logger,
            val_check_interval=1.0,
            callbacks=[AugLagrangianCallback(), early_stop_1_callback, CustomProgressBar()],
        )
        trainer.fit(
            self.model,
            DataLoader(train_dataset, batch_size=train_batch_size, num_workers=4),
            DataLoader(val_dataset, num_workers=8, batch_size=val_batch_size),
        )
        wandb.log({"nll_val": self.model.nlls_val[-1]})
        wandb.finish()

        # freeze and prune adjacency
        self.model.module.threshold()
        # WE NEED THIS BECAUSE IF it's exactly a DAG THE POWER ITERATIONS DOESN'T CONVERGE
        # TODO Just refactor and remove constraint at validation time
        self.model.module.constraint_mode = "exp"
        # remove dag constraints: we have a prediction problem now!
        self.model.gamma = 0.0
        self.model.mu = 0.0

        # Step 2:fine tune weights with frozen model
        logger = WandbLogger(project=logfile, log_model=True)
        model_name = self.model.__class__.__name__
        if do_use_polynomials and model_name == "LinearGaussianModel":
            model_name += "_poly"
        logger.experiment.config.update(
            {"model_name": model_name, "module_name": self.model.module.__class__.__name__}
        )

        early_stop_2_callback = EarlyStopping(
            monitor="Val/nll", min_delta=1e-6, patience=5, verbose=verbose, mode="min"
        )
        trainer_fine = pl.Trainer(
            num_nodes=num_gpus,
            max_epochs=num_fine_epochs,
            logger=logger,
            val_check_interval=1.0,
            callbacks=[early_stop_2_callback, CustomProgressBar()],
        )
        trainer_fine.fit(
            self.model,
            DataLoader(train_dataset, batch_size=train_batch_size),
            DataLoader(val_dataset, num_workers=2, batch_size=val_batch_size),
        )

        # Original implementation extracts the network structure for later inspection
        pred_adj = self.model.module.weight_mask.detach().cpu().numpy()
        
        assert np.equal(np.mod(pred_adj, 1), 0).all()

        # Step 4: add valid nll and dump metrics
        pred = trainer_fine.predict(
            ckpt_path="best",
            dataloaders=DataLoader(val_dataset, num_workers=8, batch_size=val_batch_size),
        )
        val_nll = np.mean([x.item() for x in pred])
    
        acyclic = int(self.model.module.check_acyclicity())
        if acyclic != 1:
            print("""
            Graph is not acyclic! Problems:
            
            - Joint density may not be well-defined, so NLL may not be a fair evaluation metric
            - Predicted expression may diverge as effects propagate in cycles (yields NaN's)
            
            Possible solutions:

            - Retrain with different inputs (e.g. more epochs)
            - Continue training where you left off by calling the 'train' method with model_type == "already_trained"
            - Avoid using likelihood, and avoid divergence by using a small 'maxiter' in the 'simulateKO()' method.
            
            """)
        wandb.log(
            {
                "val nll": val_nll,
                "acyclic": acyclic,
                "n_edges": pred_adj.sum(),
            }
        )
        
        wandb.finish()
        
        return self

    def predict(self, predictions_metadata = pd.DataFrame, baseline_expression: np.ndarray = None):
        """Predict expression after perturbation.

        Args:
            predictions_metadata (pd.DataFrame):  Dataframe with columns `cell_type`, `timepoint`, `perturbation`, and 
                `expression_level_after_perturbation`. It will default to `predictions.obs` or `starting_expression.obs` if those 
                are provided. The meaning is "predict expression in `cell_type` at `time_point` if `perturbation` were set to 
                `expression_level_after_perturbation`". Anything with expression equal tonp.nan will be treated as a control, no matter the name.
                `perturbation` and `expression_level_after_perturbation` can contain comma-separated strings for multi-gene perturbations, for 
                example "NANOG,POU5F1" for `perturbation` and "5.43,0.0" for `expression_level_after_perturbation`. 
                Anything with an unrecognized gene name or a NaN expression or both is treated like a control, e.g. [('Control', 0), ('NANOG', np.NaN)]
            baseline_expression (numpy array, optional): starting expression profile, n_perts by n_features. Defaults to None.

        Returns:
            anndata.AnnData: predicted expression
        """
        genes = self.train_dataset.dataset.adata.var_names
        assert predictions_metadata.columns.is_unique, "Column names in predictions_metadata must be unique."

        # Check or make baseline expression
        if baseline_expression is None:
            baseline_expression = np.zeros((predictions_metadata.shape[0], self.train_dataset.dataset.adata.n_vars))
            baseline_expression_one = self.train_dataset.dataset.adata.X[
                self.train_dataset.dataset.adata.obs["is_control"],:
            ].mean(axis=0)
            for i in range(baseline_expression.shape[0]):
                baseline_expression[i,:] = baseline_expression_one.copy()
        else:
            assert type(baseline_expression) in {np.ndarray, np.matrix}, f"baseline_expression must be a numpy array; got type {type(baseline_expression)}"
            assert baseline_expression.shape[0] == predictions_metadata.shape[0], f"baseline_expression must have {predictions_metadata.shape[0]} obs; got {baseline_expression.shape[0]}."
            assert baseline_expression.shape[1] == len(genes), f"baseline_expression must have {len(genes)} obs; got {baseline_expression.shape[1]}."
            
        # Make container for predictions
        predicted_adata = anndata.AnnData(
            X = np.zeros((predictions_metadata.shape[0], len(genes))),
            var = self.train_dataset.dataset.adata.var.copy(),
            obs = predictions_metadata,
            dtype = np.float64
        )

        # Parse and simulate perturbations
        def convert_gene_symbol_to_index(gene):
            return [i for i,g in enumerate(genes) if g==gene]
        
        def reformat_perturbation(p):
            # Comma-separated lists allow multi-gene perts
            pert_genes = str(p["perturbation"]).split(",")
            target_val = [float(f) for f in str(p["expression_level_after_perturbation"]).split(",")]
            assert len(pert_genes) == len(target_val), f"Malformed perturbation in sample {i}: {p}. Parsed as: {pert_genes}, {target_val}."
            # Ignore anything with unknown expression or gene 
            not_nan = [i for i, x in enumerate(target_val) if not np.isnan(x)]
            pert_genes = [pert_genes[i] for i in not_nan]
            target_val = [target_val[i] for i in not_nan]
            genes_recognized = []
            for index_of_recognized_genes, g in enumerate(pert_genes):
                if g in genes:
                    genes_recognized.append(index_of_recognized_genes)
                else:
                    print(f"Warning: Post-perturbation expression is not NaN. But because the perturbed gene {g} cannot be located, the profile is treated as a control sample lacking perturbation.")
            pert_genes = [pert_genes[i] for i in genes_recognized]
            target_val = [target_val[i] for i in genes_recognized]
            target_loc = [convert_gene_symbol_to_index(g) for g in pert_genes]
            return target_loc, target_val

        KO_gene_indices = [None for _ in predictions_metadata["perturbation"]]
        KO_gene_values  = [None for _ in predictions_metadata["perturbation"]]
        for i, p in enumerate(predictions_metadata.iterrows()):
            KO_gene_indices[i], KO_gene_values[i]  = reformat_perturbation(p[1])

        with torch.no_grad():
            predicted_adata.X  = self.model.simulateKO(
                control_expression = baseline_expression,
                KO_gene_indices    = KO_gene_indices,
                KO_gene_values     = KO_gene_values,
                maxiter            = 1,
            )
        
        return predicted_adata

class DCDFGCV:
    def __init__(self):
        self.final_model = None
        return
    
    def train(
        self,
        adata: anndata.AnnData, 
        regularization_parameters: np.ndarray = None,
        latent_dimensions: np.ndarray = None,
        **kwargs,
    ):
        if regularization_parameters is None: 
            regularization_parameters = np.array([10, 1, 0.1, 0.01, 0.001])
        if latent_dimensions is None:
            latent_dimensions = np.array([5, 10, 20, 50])
        # These never pass!! Why not??
        if "regularization_parameter" in kwargs.keys():
            raise ValueError("regularization_parameter may not be passed to DCDFGCV. Try DCDFGWrapper instead.")
        if "num_modules" in kwargs.keys():
            raise ValueError("num_modules may not be passed to DCDFGCV. Try DCDFGWrapper instead.")
        data_splitter = KFold(3)
        self.error = {r:{l:0.0 for l in latent_dimensions} for r in regularization_parameters}
        for r in regularization_parameters:
            for l in latent_dimensions:
                for train,test in data_splitter.split(adata.X):
                    obs = np.where(adata.obs["is_control"]) # training data must include unperturbed samples
                    train = np.array(list(train) + list(obs[0]))
                    factor_graph_model = DCDFGWrapper()
                    try:
                        factor_graph_model.train(
                            adata = adata[train,:],
                            regularization_parameter = r,
                            num_modules=l,
                            **kwargs,
                        )
                        predicted = factor_graph_model.predict(
                            predictions_metadata = adata[test,:].obs[["perturbation", "expression_level_after_perturbation"]]
                        )
                        self.error[r][l] += np.linalg.norm(predicted.X - adata[test,:].X)
                    except Exception as e:
                        print(f"DCD-FG failed with error {repr(e)} on params r={r}, l={l}.")
                        self.error[r][l] = np.Inf
                    break

        # Find min error 
        min = np.inf
        self.best_regularization_parameter = 10
        self.best_dimension = 5
        for r in regularization_parameters:
            for l in latent_dimensions:
                if self.error[r][l] < min:
                    min = self.error[r][l]
                    self.best_regularization_parameter = r
                    self.best_dimension = l

        print(f"best dimension: {self.best_dimension}")
        print(f"best regularization parameter: {self.best_regularization_parameter}")
        print(f"best error: {min}")
        print("all error:")
        print(self.error)
        self.final_model = DCDFGWrapper()
        try:
            self.final_model.train(
                adata = adata,
                regularization_parameter = self.best_regularization_parameter,
                num_modules = self.best_dimension,
                **kwargs,
            )
        except Exception as e:
            raise ValueError(f"Unable to train DCDFG even using the best parameters we could find. Error was {repr(e)}.")
        return self.final_model