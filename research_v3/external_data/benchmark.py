"""Importable, side-effect-free benchmark API over frozen local artifacts.

Import performs no downloads, training or file writes. Explicit evaluation
restores temporary evaluator overrides even if model fitting fails. The core
uses module-level configuration, so concurrent evaluation in one Python process
is unsupported; separate worker processes are safe.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import sys
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO))
from final_solution.training import core_experiment as core
from final_solution.training import train_and_evaluate as evaluator

TREASURY_LAG7_FEATURES=[f"treasury_{symbol}_stale_{suffix}" for symbol in ("t10yie","t5yifr") for suffix in ("level","chg5","chg20")]
BASIS_FEATURES=core.CORE_FEATURES+core.VOL_FEATURES+["moex_cny_close_minus_fixing_same_session"]

def load_frozen_panel() -> pd.DataFrame:
    """Explicit disk read; includes all ablation feature columns."""
    return pd.read_parquet(HERE / "feature_panel.parquet")

def augment_panel(panel:pd.DataFrame,features=TREASURY_LAG7_FEATURES) -> pd.DataFrame:
    """Backward-built features joined by exact date/corridor, never backfilled.

    The frozen feature panel starts in 2020. Earlier feature values stay missing
    on an extended official-history panel; this is not 10 years of full macro
    coverage and must be reported when evaluating long train windows.
    """
    features=list(features)
    right=load_frozen_panel()[["date","corridor",*features]]
    return panel.drop(columns=[c for c in features if c in panel],errors="ignore").merge(right,on=["date","corridor"],how="left",validate="one_to_one").sort_values(["date","corridor"]).reset_index(drop=True)

@contextmanager
def _settings(feature_names,config_id,train_years,disable_early_stopping):
    old_years=core.TRAIN_WINDOW_YEARS
    old_make=core.make_model
    existed=config_id in core.FEATURE_GROUPS
    old_features=core.FEATURE_GROUPS.get(config_id)
    core.TRAIN_WINDOW_YEARS=train_years
    core.FEATURE_GROUPS[config_id]=list(feature_names)
    if disable_early_stopping:
        def factory(kind,features):
            model=old_make(kind,features)
            model.named_steps["classifier"].set_params(early_stopping=False)
            return model
        core.make_model=factory
    try: yield
    finally:
        core.TRAIN_WINDOW_YEARS=old_years
        core.make_model=old_make
        if existed:core.FEATURE_GROUPS[config_id]=old_features
        else:core.FEATURE_GROUPS.pop(config_id,None)

def evaluate(panel:pd.DataFrame,feature_names,config_id:str,*,train_years=2,years=(2023,2024,2025,2026),horizon=5,disable_early_stopping=False):
    """Return (core prediction rows, core fold/corridor metrics).

    Uses a full prior validation year and the existing calibration/cadence
    policy. For root long-window experiments pass disable_early_stopping=True:
    sklearn's automatic stopping otherwise changes after 10,000 train rows.
    Nothing is persisted unless the caller writes the returned frames.
    """
    targeted=core.add_target(panel,horizon)
    preds=[];rows=[]
    with threadpool_limits(limits=1),_settings(feature_names,config_id,train_years,disable_early_stopping):
        for year in years:
            spec=core.Experiment(config_id,"hist_gradient_boosting",config_id)
            r,p,_=core.run_fold(spec,horizon,year,targeted)
            p["model_kind"]=spec.model_kind;p["feature_set"]=spec.feature_set
            preds.append(p);rows.extend(r)
    return pd.concat(preds,ignore_index=True),pd.DataFrame(rows)

def score(predictions:pd.DataFrame,fold_metrics:pd.DataFrame) -> pd.DataFrame:
    return evaluator.merge_predictive_and_policy(core,fold_metrics,predictions)
