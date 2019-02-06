import numpy as np
import pandas as pd

from model_classes import *


# all the code in this file is referred to the toy example in the paper "Didelot, Xavier, et al. "Likelihood-free
# estimation of model evidence." Bayesian analysis 6.1 (2011): 49-76."


# ABC_MC ALGORITHM

def ABC_MC(models, n_samples, n_sim):
    # implement the ABC-MC using a deterministic model choice (meaning we assume uniform prior probability over 
    # the models)

    # models is a list of object of class model, as defined below 

    # NB: this only computes the reference table

    n_models = len(models)
    results = pd.DataFrame(columns=("model", "param", "s_1", "t_1"))

    for model_number in range(n_models):
        for i in range(n_sim // n_models):
            mod = models[model_number](n_samples)  # instantiate the model
            res = mod.gen_sample()
            results = results.append(res.append(pd.Series(model_number, index=["model"])), ignore_index=True)

    return results


def select_relevant_simulations(results, reference_statistics, quantile=0.05, use_all_statistics=True,
                                distance="euclidean"):
    # results is likely the output of the previous function
    # this function automatically selects the threshold for the distance that leads to the selection of a given
    # quantile.
    # reference_statistics is a pd.Series or dictionary containing are the statistics with respect to which the
    # (Euclidean) distance is computed
    # if all_statistics=True, the distance is computed using both statistics (s_1 and t_1, as defined in the paper),
    # eitherwise using only the s_1
    # distance can be either "euclidean" or "chebyshev"

    # first step: compute all the distances:    
    if use_all_statistics:
        if distance == "euclidean":
            results["distance"] = np.sqrt((results["s_1"] - reference_statistics["s_1"]) ** 2 + \
                                          (results["t_1"] - reference_statistics["t_1"]) ** 2)
        elif distance == "chebyshev":
            # this is the Chebyshev distance
            results["distance"] = np.max(np.stack((np.abs((results["s_1"] - reference_statistics["s_1"])),
                                                   np.abs((results["t_1"] - reference_statistics["t_1"])))), axis=0)
    else:
        results["distance_partial"] = np.abs(results["s_1"] - reference_statistics["s_1"])

    # second step: sort the dataframe according to the distance and return only the smallest quantile. 
    if use_all_statistics:
        return results.sort_values(by="distance").iloc[0:np.int(quantile * len(results))]
    else:
        return results.sort_values(by="distance_partial").iloc[0:np.int(quantile * len(results))]


def select_relevant_simulations_threshold(results, reference_statistics, threshold=0.05, use_all_statistics=True,
                                          distance="euclidean"):
    # results is likely the output of the previous function
    # this function automatically uses a fixed threshold for selecting the relevant simulations
    # reference_statistics is a pd.Series or dictionary containing are the statistics with respect to which the
    # (Euclidean) distance is computed
    # if all_statistics=True, the distance is computed using both statistics (s_1 and t_1, as defined in the paper),
    # eitherwise using only the s_1
    # distance can be either "euclidean" or "chebyshev"

    # first step: compute all the distances:    
    if use_all_statistics:
        if distance == "euclidean":
            results["distance"] = np.sqrt((results["s_1"] - reference_statistics["s_1"]) ** 2 + \
                                          (results["t_1"] - reference_statistics["t_1"]) ** 2)
        elif distance == "chebyshev":
            # this is the Chebyshev distance
            results["distance"] = np.max(np.stack((np.abs((results["s_1"] - reference_statistics["s_1"])),
                                                   np.abs((results["t_1"] - reference_statistics["t_1"])))), axis=0)
    else:
        results["distance_partial"] = np.abs(results["s_1"] - reference_statistics["s_1"])
        # second step: sort the dataframe according to the distance and return only the smallest quantile.
    if use_all_statistics:
        return results[results["distance"] < threshold]
    else:
        return results[results["distance_partial"] < threshold]


def estimate_BF_given_reference_table(reference_table, datasets_summary, quantile=0.005):
    # this function uses the same reference table to estimate BF for all datasets present in datasets_summary

    n_sim = reference_table.shape[0]

    print("Number of simulations that are retained from the reference table:", quantile * n_sim)

    n_datasets = datasets_summary.shape[0]

    for i in range(n_datasets):
        dataset_res = datasets_summary.iloc[i]
        ref_stats = {"s_1": dataset_res["s_1"], "t_1": dataset_res["t_1"]}

        # compute distances and take smallest quantile, for all statistics:  
        res_filtered_both = select_relevant_simulations(reference_table, ref_stats, quantile=quantile,
                                                        use_all_statistics=True)

        # print(ref_stats,"\n",  res_filtered.iloc[0], "\n")

        try:
            BF = sum(res_filtered_both["model"] == 0) / sum(res_filtered_both["model"] == 1)
        except:
            BF = np.infty

        datasets_summary.iloc[i]["appr_BF_all_stats"] = BF

        # same for only s_1

        res_filtered = select_relevant_simulations(reference_table, ref_stats, quantile=quantile,
                                                   use_all_statistics=False)

        # print(ref_stats,"\n",  res_filtered.iloc[0], "\n")

        try:
            BF = sum(res_filtered["model"] == 0) / sum(res_filtered["model"] == 1)
        except:
            BF = np.infty

        datasets_summary.iloc[i]["appr_BF_1_stat"] = BF

    return datasets_summary


# UTILITIES FUNCTIONS APPLYING THE ABOVE CLASSES:

def true_BF(dataset):
    # compute the BF of the Poisson model with respect to the Geometric one. 

    # first model
    poi_mod = PoissonModel(n_samples=len(dataset))
    poi_mod.set_obs(dataset)
    poi_mod.evidence()

    # second model
    geo_mod = GeometricModel(n_samples=len(dataset))
    geo_mod.set_obs(dataset)
    geo_mod.evidence()

    log_BF = poi_mod.evidence() - geo_mod.evidence()
    return np.exp(log_BF)


def compute_reference_statistics(dataset):
    # compute the two reference statistics s_1 and t_1 as defined in the paper. 

    poi_mod = PoissonModel(n_samples=len(dataset))
    poi_mod.set_obs(dataset)

    return {"s_1": poi_mod.s_1, "t_1": poi_mod.t_1}


def approximate_BF(n_samples, reference_statistics, n_sim=30000, models=(PoissonModel, GeometricModel),
                   quantile=0.05):
    # this computes the approximate BF given the n_samples of the original dataset and the reference statistics,
    # applying the above functions

    # generate the reference table (expensive step)
    res = ABC_MC(models, n_samples, n_sim)

    # compute distances and pick smallest quantile, with all statistics
    res_filtered = select_relevant_simulations(res, reference_statistics, quantile=quantile,
                                               use_all_statistics=True)

    try:
        BF_all_statistics = sum(res_filtered["model"] == 0) / sum(res_filtered["model"] == 1)
    except:
        BF_all_statistics = np.infty

    # compute distances and pick smallest quantile, with only 1 statistic
    res_filtered = select_relevant_simulations(res, reference_statistics, quantile=quantile,
                                               use_all_statistics=False)

    try:
        BF_1_stat = sum(res_filtered["model"] == 0) / sum(res_filtered["model"] == 1)
    except:
        BF_1_stat = np.infty

    return BF_all_statistics, BF_1_stat
