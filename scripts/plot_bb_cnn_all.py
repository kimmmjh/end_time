#!/usr/bin/env python3
"""Plot the latest completed result per CNN configuration; no inference."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/bb_cnn_all_matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/bb_cnn_all_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'results/analysis'
PLOTS = ROOT / 'results/plots/bb/code_capacity/tanner_cnn'
SOURCES = ('2026_09_27', '2026_10_10', '2026_10_12')
COLORS = {1: '#8c7a65', 2: '#8b63b6', 3: '#179d91', 4: '#3478c6', 5: '#d45238'}


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def load_results():
    rows = []
    for date in SOURCES:
        source_rows = read_csv(ANALYSIS / f'bb_update_{date}_cnn_final.csv')
        for r in source_rows:
            code, p, depth, seed = r['code'], float(r['p']), int(r['depth']), int(r['seed'])
            epochs = int(r.get('to_epochs', r.get('epochs')))
            history_path = ROOT / r['source']
            if not history_path.is_file():
                # Some 100-epoch sources were retained at the repo root for transfer.
                parts = Path(r['source']).parts
                job_index = next(i for i, part in enumerate(parts) if part.startswith('resdir_'))
                history_path = ROOT.joinpath(*parts[job_index:])
            history = json.loads(history_path.read_text())
            assert len(history['train']) == epochs
            assert history['config']['cnn_depth'] == depth
            assert history['final']['shots'] == int(r['shots']) == 65536
            row = dict(campaign=date, code=code, p=p, depth=depth, seed=seed, epochs=epochs,
                       primary=seed == int(code[2:])*100000 + 1000 + round(p*1000),
                       recovery=r.get('recovery') == 'True', shots=int(r['shots']),
                       source=str(history_path.relative_to(ROOT)))
            for method in ('raw', 'osd'):
                rate, count = float(r[f'{method}_ler']), int(r[f'{method}_failures'])
                assert abs(rate - count/row['shots']) < 1e-12
                assert abs(rate - history['final'][method]['logical_error_rate']) < 1e-12
                low = float(r.get(f'{method}_ci95_low', r.get(f'{method}_ler_low')))
                high = float(r.get(f'{method}_ci95_high', r.get(f'{method}_ler_high')))
                assert 0 <= low <= rate + 1e-15 and rate <= high <= 1
                row.update({f'{method}_ler': rate, f'{method}_failures': count,
                            f'{method}_low': low, f'{method}_high': high})
            rows.append(row)
    assert len(rows) == 67 and sum(r['recovery'] for r in rows) == 1
    assert len({r['source'] for r in rows}) == len(rows)
    return rows


def latest_results(rows):
    latest = {}
    for row in rows:
        key = (row['code'], row['p'], row['depth'], row['seed'])
        previous = latest.get(key)
        if previous is None or (row['epochs'], row['campaign']) > (previous['epochs'], previous['campaign']):
            latest[key] = row
    return sorted(latest.values(), key=lambda r: (r['code'], r['depth'], r['p'], r['seed']))


def draw_point(ax, r, method, color, marker, alpha=1., size=5):
    x, y = r['p']*100, r[f'{method}_ler']*100
    low, high = r[f'{method}_low']*100, r[f'{method}_high']*100
    if y == 0:
        # A zero rate has no position on a log axis; display only its upper bound.
        ax.plot(x, high, marker='v', ms=7, mfc='white', mec=color, alpha=alpha, zorder=5)
    else:
        ax.errorbar(x, y, yerr=[[y-low], [high-y]], fmt=marker, color=color,
                    mfc='white' if r['recovery'] else color, ms=size,
                    capsize=2, elinewidth=.8, alpha=alpha, zorder=4)


def main():
    all_rows = load_results()
    rows = latest_results(all_rows)
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plotted = 0
    for i, code in enumerate(('bb72', 'bb144')):
        for j, method in enumerate(('raw', 'osd')):
            ax = axes[i, j]
            for depth, color in COLORS.items():
                group = [r for r in rows if r['code']==code and r['depth']==depth]
                primary = {r['p']:r for r in group if r['primary']}
                grid = (.02, .04, .06, .08)
                ys = [primary[p][f'{method}_ler']*100 if p in primary and primary[p][f'{method}_ler'] > 0 else np.nan for p in grid]
                ax.plot(np.array(grid)*100, ys, ls='-', lw=2, color=color)
                for r in group:
                    draw_point(ax, r, method, color,
                               '*' if r['recovery'] else 'o' if r['primary'] else 'x',
                               alpha=1. if r['primary'] else .5, size=12 if r['recovery'] else 5)
                    plotted += 1
            if method == 'raw':
                refs = [r for r in read_csv(ANALYSIS/'bb_update_2026_09_27_capacity_references.csv')
                        if r['code']==code and r['method']=='Neural BP4 T12']
                refs.sort(key=lambda r: float(r['p']))
                ax.plot([float(r['p'])*100 for r in refs], [float(r['ler'])*100 for r in refs],
                        color='#444444', ls='-.', marker='+', lw=1.3, alpha=.8)
            ax.set(title=f'{code.upper()}  |  {"CNN alone" if method=="raw" else "CNN + residual OSD-0"}',
                   xlabel='Physical error rate p (%)', ylabel='Block LER (%)', yscale='log',
                   xticks=[2,4,6,8], xlim=(1.75,8.25))
            ax.grid(alpha=.2, which='major')
            panel = [r for r in rows if r['code']==code]
            lower = min(r[f'{method}_low'] if r[f'{method}_ler'] else r[f'{method}_high'] for r in panel)*100
            upper = max(r[f'{method}_high'] for r in panel)*100
            ax.set_ylim(lower*.6, min(110,upper*1.6))
            if code=='bb144':
                rec=next(r for r in rows if r['code']==code and r['recovery'])
                ax.annotate('400-epoch recovery', xy=(4,rec[f'{method}_ler']*100),
                            xytext=(8, -23 if method=='raw' else 15), textcoords='offset points',
                            fontsize=8, color=COLORS[5], arrowprops={'arrowstyle':'-', 'color':COLORS[5]})
    assert plotted == 2*len(rows), (plotted,len(rows))
    depth_handles=[Line2D([],[],color=c,lw=2,label=f'Depth {d}') for d,c in COLORS.items()]
    fig.legend(handles=depth_handles, loc='lower center', bbox_to_anchor=(.5,.075),ncol=5,frameon=False)
    budget_handles = [Line2D([],[],color=COLORS[5],marker='*',ls='',ms=10,label='BB144 depth 5, p=4%: 400 epochs'),
                       Line2D([],[],color='#666666',marker='x',ls='',label='Additional seeds'),
                       Line2D([],[],color='#444444',ls='-.',marker='+',label='Prior Neural BP4 (raw panels)')]
    fig.legend(handles=budget_handles,loc='lower center',bbox_to_anchor=(.5,.028),ncol=3,frameon=False,fontsize=9)
    fig.suptitle(f'Tanner CNN: latest completed results only\nDepolarizing code capacity · {len(rows)} configurations · earlier training stages omitted',fontsize=16,y=.975)
    fig.subplots_adjust(left=.075,right=.98,top=.895,bottom=.17,wspace=.23,hspace=.31)
    destination=PLOTS/'overview.png'
    fig.savefig(destination,dpi=190,facecolor='white')
    plt.close(fig)
    for name, values in (('bb_cnn_all_final.csv', all_rows), ('bb_cnn_latest_final.csv', rows)):
        with (ANALYSIS/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    print(f'Plotted {len(rows)} latest models ({plotted} raw/OSD points), omitted {len(all_rows)-len(rows)} earlier stages: {destination}')
    print(f'Underlying values and original history paths: {ANALYSIS / "bb_cnn_latest_final.csv"}')


if __name__=='__main__':
    main()
