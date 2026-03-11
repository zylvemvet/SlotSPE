#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: visual_utils.py
@author:zyl
@contact:yilan.zhang@kaust.edu.sa
@time:12/25/24 3:14 PM
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import restricted_mean_survival_time
from numpy.random import default_rng
import pickle
import pandas as pd

def _load_pkl(filename):
    loader = open(filename,'rb')
    file = pickle.load(loader)
    loader.close()
    return file



def _draw_kaplan_meier(event_low, time_low, event_high, time_high, title, save_path):
    '''
    Draw the kaplan meier curve
    :param event_low: low risk event
    :param time_low: low risk time
    :param event_high: high risk event
    :param time_high: high risk time
    :param title: title of the figure
    :param save_path: save path of the figure
    :return:
    '''
    ax = plt.subplot(111)

    kmf = KaplanMeierFitter()
    #----> low risk
    time_low = np.array(time_low)
    event_low = np.array(event_low)
    kmf.fit(time_low, event_low, label='low risk')
    ax = kmf.plot_survival_function(ax=ax,ci_show=True,show_censors=True)
    # x_low, y_low = kaplan_meier_estimator(event_low, time_low)
    # plt.step(x_low, y_low,where="post", label="low risk",color='green')
    # plt.plot(x_low, y_low,'+',color='green')

    #----> high risk
    time_high = np.array(time_high)
    event_high = np.array(event_high)
    kmf.fit(time_high, event_high, label='high risk')
    ax = kmf.plot_survival_function(ax=ax,ci_show=True,show_censors=True)
    # x_high, y_high = kaplan_meier_estimator(event_high, time_high)
    # plt.step(x_high, y_high,where="post", label="high risk",color='red')
    # plt.plot(x_high, y_high,'+',color='red')

    plt.ylim(0, 1.1)

    plt.ylabel("est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")

    #----> logrank test
    p = logrank_test(time_low, time_high, event_low, event_high)
    title = title + ' p-value: {:.4e}'.format(p.p_value)
    plt.title(title)
    plt.legend(loc="best")
    plt.savefig(save_path)
    # plt.show()
    # plt.close()


def _process_results_km(path,folds=5):
    '''
    :param path: the path of the pkl file
    :param folds: how many folds
    :return:
    '''
    total_results_dict = {}
    low_risk_results = []
    for i in range(folds):
        pkl_path = os.path.join(path, 'split_{}_results.pkl'.format(i))
        results = _load_pkl(pkl_path)
        total_results_dict = {**total_results_dict, **results}
    total_results_dict = sorted(total_results_dict.items(), key=lambda x: x[1]['risk'], reverse=False)
    middle = int(len(total_results_dict)/2)
    low_risk_results = total_results_dict[:middle]
    high_risk_results = total_results_dict[middle:]

    event_time_low = []
    event_time_high = []
    censorships_low = []
    censorships_high = []
    risk_score_low = []
    risk_score_high = []
    for i in range(len(low_risk_results)):
        event_time_low.append(low_risk_results[i][1]['time'])
        censorships_low.append(low_risk_results[i][1]['censor'])
        risk_score_low.append(low_risk_results[i][1]['risk'])
    for i in range(len(high_risk_results)):
        event_time_high.append(high_risk_results[i][1]['time'])
        censorships_high.append(high_risk_results[i][1]['censor'])
        risk_score_high.append(high_risk_results[i][1]['risk'])
    event_low = (1- np.array(censorships_low)).astype(bool)
    event_high = (1- np.array(censorships_high)).astype(bool)

    # event_time_high = np.array([total_results_dict[i][1]['time'] / 12 for i in range(len(total_results_dict))]).max()
    # event_time_low = np.array([total_results_dict[i][1]['time'] / 12 for i in range(len(total_results_dict))]).min()

    event_time_low = np.array(event_time_low)
    event_time_high = np.array(event_time_high)

    # risk_score_low = np.array(risk_score_low)
    # risk_score_high = np.array(risk_score_high)

    # normalize risk score
    # risk_score_low_years = (risk_score_low - risk_score_low.min()) / (risk_score_high.max() - risk_score_low.min())*\
    #                  (event_time_high - event_time_low) + event_time_low
    # risk_score_high_years = (risk_score_high - risk_score_low.min()) / (risk_score_high.max() - risk_score_low.min())*\
    #                     (event_time_high - event_time_low) + event_time_low

    _draw_kaplan_meier(event_low, event_time_low, event_high, event_time_high, title='Kaplan-Meier', save_path=os.path.join(path, 'Kaplan-Meier.png'))
    # _draw_kaplan_meier(event_low, risk_score_low_years, event_high, risk_score_high_years, title='Kaplan-Meier', save_path=os.path.join(path, 'Kaplan-Meier.png'))



    return results

def _draw_quantile_or_flag(ax, km, color, q=0.5, label="Median"):
    """
    Draw KM quantile if it exists; otherwise annotate 'Not reached'.
    """
    # Try lifelines' quantile (works like median at q=0.5)
    try:
        # lifelines >= 0.26 has km.quantile(q); older has median_survival_time_
        med = float(km.quantile(q))  # returns np.nan if not reached
    except Exception:
        med = km.median_survival_time_ if q == 0.5 else np.nan

    if np.isfinite(med):
        ax.axhline(1 - q, linestyle="--", linewidth=1, alpha=0.6, color="0.35")
        ax.axvline(med, linestyle="--", linewidth=1, alpha=0.8, color=color)
        ax.text(med, 1 - q + 0.02, f"{label}={med:.1f}", color=color,
                ha="left", va="bottom", fontsize=max(16, plt.rcParams['font.size']-1))
    else:
        # Put a small note near the right edge
        x_right = ax.get_xlim()[1]
        ax.text(0.98, 0.06, f"{label} not reached",
                transform=ax.transAxes, ha="right", va="bottom",
                color=color, fontsize=max(10, plt.rcParams['font.size']-1),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor=color))

def _auto_tau(km_low, km_high, time_low, time_high, strategy="min_common_support", min_at_risk=10):
    # Strategy 1: last time both groups have >= min_at_risk
    if strategy == "min_common_support":
        tL = km_low.event_table.index.values
        tH = km_high.event_table.index.values
        t_common = np.intersect1d(tL, tH)
        if t_common.size:
            arL = km_low.event_table.loc[t_common, "at_risk"].values
            arH = km_high.event_table.loc[t_common, "at_risk"].values
            mask = (arL >= min_at_risk) & (arH >= min_at_risk)
            if mask.any():
                return float(t_common[mask].max())
    # Fallback: min of max follow-up
    return float(min(np.max(time_low), np.max(time_high)))

def _fill_rmst_area(ax, km, color, tau, alpha=0.15):
    sf = km.survival_function_
    x = sf.index.values
    y = sf.iloc[:, 0].values
    # Ensure start at 0 and end at tau
    x = np.r_[0.0, x, tau]
    y = np.r_[1.0, y, float(km.predict(tau))]
    ax.fill_between(x, y, step="post", where=x <= tau, color=color, alpha=alpha)
    ax.axvline(tau, color="0.35", ls=":", lw=1)

def _rmst_km(time, event, tau):
    km = KaplanMeierFitter().fit(time, event_observed=event)
    return restricted_mean_survival_time(km, t=tau)

def _rmst_compare(time_low, event_low, time_high, event_high, tau, n_boot=1000, seed=0, return_ratio=True):
    rng = np.random.default_rng(seed)
    idxL = np.arange(len(time_low)); idxH = np.arange(len(time_high))
    rmstL = _rmst_km(time_low, event_low, tau)
    rmstH = _rmst_km(time_high, event_high, tau)

    diffs = []
    ratios = []
    for _ in range(n_boot):
        bL = rng.choice(idxL, size=len(idxL), replace=True)
        bH = rng.choice(idxH, size=len(idxH), replace=True)
        rL = _rmst_km(time_low[bL],  event_low[bL],  tau)
        rH = _rmst_km(time_high[bH], event_high[bH], tau)
        diffs.append(rH - rL)
        if return_ratio:
            ratios.append((rH + 1e-12) / (rL + 1e-12))
    diffs = np.array(diffs)
    ci_diff = np.percentile(diffs, [2.5, 97.5])
    p_diff = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())

    out = {
        "tau": tau,
        "low":  {"rmst": rmstL},
        "high": {"rmst": rmstH},
        "delta": {"diff": rmstH - rmstL, "ci": (ci_diff[0], ci_diff[1]), "p": float(p_diff)},
    }

    if return_ratio:
        ratios = np.array(ratios)
        log_ci = np.percentile(np.log(ratios + 1e-12), [2.5, 97.5])
        out["ratio"] = {
            "value": (rmstH + 1e-12) / (rmstL + 1e-12),
            "ci": (float(np.exp(log_ci[0])), float(np.exp(log_ci[1])))
        }
    return out


def draw_kaplan_meier(
    event_low, time_low,
    event_high, time_high,
    title="Kaplan–Meier",
    save_path=None,
    show=True,
    figsize=(7.5, 6),
    dpi=300,
    colors=("tab:blue", "tab:red"),
    ci_alpha=0.20,
    grid=True,
    show_medians=True,
    legend_loc="lower left",
    base_fontsize=16,   # <-- control everything here
):
    """
    Kaplan–Meier plot with larger fonts.
    """
    # --- Prepare data
    time_low  = np.asarray(time_low)
    event_low = np.asarray(event_low).astype(bool)
    time_high  = np.asarray(time_high)
    event_high = np.asarray(event_high).astype(bool)

    # --- Fit models
    km_low  = KaplanMeierFitter(label="Low risk")
    km_high = KaplanMeierFitter(label="High risk")
    km_low.fit(durations=time_low, event_observed=event_low)
    km_high.fit(durations=time_high, event_observed=event_high)

    # --- Log-rank test
    lr = logrank_test(time_low, time_high, event_observed_A=event_low, event_observed_B=event_high)

    # --- Figure
    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if grid:
        ax.grid(True, which="major", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # --- Plot KM
    km_low.plot_survival_function(
        ax=ax, ci_show=True, ci_alpha=ci_alpha, color=colors[0],
        show_censors=True, censor_styles={"ms": 7, "marker": "+"}
    )
    km_high.plot_survival_function(
        ax=ax, ci_show=True, ci_alpha=ci_alpha, color=colors[1],
        show_censors=True, censor_styles={"ms": 7, "marker": "+"}
    )

    # --- Median survival lines
    if show_medians:
        _draw_quantile_or_flag(ax, km_low, colors[0], q=0.5, label="Median")
        _draw_quantile_or_flag(ax, km_high, colors[1], q=0.5, label="Median")

    # --- Labels & limits
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Time $t$", fontsize=base_fontsize)
    ax.set_ylabel(r"Estimated survival $\hat{S}(t)$", fontsize=base_fontsize)

    # --- Title with p-value
    pstr = f"{lr.p_value:.2e}"
    ax.set_title(title + f" (p={pstr})", pad=15, fontsize=base_fontsize+2, weight="bold")

    # --- Legend
    leg = ax.legend(frameon=False, loc=legend_loc, fontsize=base_fontsize-2)

    # --- RMST (enhanced)
    # 1) Choose tau robustly (you can override via a function arg if you like)
    # tau_resolved = _auto_tau(km_low, km_high, time_low, time_high,
    #                          strategy="min_common_support", min_at_risk=10)
    tau_resolved = 60

    # 2) Shade area under S(t) up to tau for both groups
    _fill_rmst_area(ax, km_low, colors[0], tau_resolved, alpha=0.18)
    _fill_rmst_area(ax, km_high, colors[1], tau_resolved, alpha=0.18)

    # 3) Compute RMST, CIs, Δ, p (and ratio CI)
    res = _rmst_compare(time_low, event_low, time_high, event_high,
                        tau=tau_resolved, n_boot=1500, seed=42, return_ratio=True)

    # 4) Nicely formatted annotation box (units-aware)
    units = "months"  # <- change to your unit
    dec = 1  # decimals for printing
    # txt = (f"RMST\u209C={res['tau']:.0f} {units}\n"
    #        f"Low={res['low']['rmst']:.{dec}f}; "
    #        f"High={res['high']['rmst']:.{dec}f}\n"
    #        f"\u0394(High−Low)={res['delta']['diff']:.{dec}f} "
    #        f"[{res['delta']['ci'][0]:.{dec}f}, {res['delta']['ci'][1]:.{dec}f}], "
    #        f"p={res['delta']['p']:.3f}\n"
    #        f"Ratio={res['ratio']['value']:.2f} "
    #        f"[{res['ratio']['ci'][0]:.2f}, {res['ratio']['ci'][1]:.2f}]")

    txt = (f"\u0394(High−Low)={res['delta']['diff']:.{dec}f} \n"
           f"Ratio={res['ratio']['value']:.2f} ")

    pad = 0.02  # 2% inward from top-right
    ax.text(1 - pad, 1 - pad, txt,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=max(12, int(base_fontsize+2)),
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.9, edgecolor="0.8"),
            zorder=3)

    # --- Tick label sizes
    ax.tick_params(axis="both", labelsize=base_fontsize-2)

    # --- Save/Show
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def process_results_km(path,folds=5):
    '''
    :param path: the path of the pkl file
    :param folds: how many folds
    :return:
    '''
    '''version 1.0'''
    # total_results_dict = {}
    
    # for i in range(folds):
    #     pkl_path = os.path.join(path, 'split_{}_results.pkl'.format(i))
    #     results = _load_pkl(pkl_path)
    #     total_results_dict = {**total_results_dict, **results}

    '''version 2.0'''
    total_results_dict = {}

    for i in range(folds):
        pkl_path = os.path.join(path, f"split_{i}_results.pkl")
        if not os.path.exists(pkl_path):
            print(f"⚠️ Missing {pkl_path}, skip.")
            continue

        results = _load_pkl(pkl_path)

        # --- 提取每折 risk ---
        risk_values = np.array([v["risk"] for v in results.values()])
        mean_r, std_r = risk_values.mean(), risk_values.std()

        if std_r < 1e-8:
            print(f"⚠️ Fold {i}: std too small, skip normalization")
            std_r = 1.0

        # --- Z-score normalization ---
        # for pid, vals in results.items():
        #     vals["risk"] = (vals["risk"] - mean_r) / std_r

        # MinMax: (risk - min)/(max-min)
        min_r, max_r = risk_values.min(), risk_values.max()
        for pid, vals in results.items():
            vals["risk"] = (vals["risk"] - min_r) / (max_r - min_r + 1e-8)

        # --- merge ---
        total_results_dict.update(results)

    print(f"✅ Aggregated {len(total_results_dict)} samples across {folds} folds (risk normalized).")

    
    low_risk_results = []
    total_results_dict = sorted(total_results_dict.items(), key=lambda x: x[1]['risk'], reverse=False)
    middle = int(len(total_results_dict)/2)
    low_risk_results = total_results_dict[:middle]
    high_risk_results = total_results_dict[middle:]

    event_time_low = []
    event_time_high = []
    censorships_low = []
    censorships_high = []
    risk_score_low = []
    risk_score_high = []
    for i in range(len(low_risk_results)):
        event_time_low.append(low_risk_results[i][1]['time'])
        censorships_low.append(low_risk_results[i][1]['censor'])
        risk_score_low.append(low_risk_results[i][1]['risk'])
    for i in range(len(high_risk_results)):
        event_time_high.append(high_risk_results[i][1]['time'])
        censorships_high.append(high_risk_results[i][1]['censor'])
        risk_score_high.append(high_risk_results[i][1]['risk'])
    
    event_low = (1- np.array(censorships_low)).astype(bool)
    event_high = (1- np.array(censorships_high)).astype(bool)


    event_time_low = np.array(event_time_low)
    event_time_high = np.array(event_time_high)


    draw_kaplan_meier(event_low, event_time_low, event_high, event_time_high, title='Kaplan-Meier', save_path=os.path.join(path, 'KaplanMeier-r1.png'))


    return results
if __name__ == "__main__":
    # PATH = "/home/zhany0x/Documents/experiment/Survival/"
    path = "/home/zhany0x/Documents/experiment/Survival/kirc/SlotNetv8/0.0005_b32_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_64_rG_16_sp_new"
    # data_path = os.path.join(PATH, path)
    process_results_km(path, folds=5)
