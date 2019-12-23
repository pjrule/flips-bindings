"""Generates basic figures and statistics for the first set of experiments."""
import os
import sys
import json
import click
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm
from gerrychain import Graph, Election, GeographicPartition, updaters
from gerrychain.graph.geo import reprojected
from recom import RecomChain
from scipy.signal import resample


def weighted_downsample_1d(plans, data_filter, resolution=10000):
    n_points = sum(plan.multiplicity for plan in plans)
    raw_signal = np.zeros(n_points)
    idx = 0
    for plan in plans:
        raw_signal[idx:idx + plan.multiplicity] = data_filter(plan)
        idx += plan.multiplicity
    resampled_signal = resample(raw_signal, resolution)
    resampled_t = np.linspace(0, n_points, resolution)
    return resampled_t, resampled_signal


def cut_edges_fig(output_dir, prefix, normal, reversible):
    """Generates the normal vs. reversible ReCom cut edges figure."""
    n_edges = len(normal[0].partition.graph.edges)

    def rel_cut_edges(step, n_edges=n_edges):
        return len(step.partition['cut_edges']) / n_edges

    normal_t, normal_rel_cut_edges = weighted_downsample_1d(normal,
                                                            rel_cut_edges)
    reverse_t, reverse_rel_cut_edges = weighted_downsample_1d(reversible,
                                                              rel_cut_edges)
    plt.plot(normal_t, normal_rel_cut_edges, label='ReCom')
    plt.plot(reverse_t, reverse_rel_cut_edges, label='Reversible ReCom')
    plt.legend()
    plt.ylim(0, 0.2)  # TODO: make upper bound dynamic
    plt.xlabel('Steps')
    plt.ylabel('Proportion Cut Edges')
    if prefix:
        filename = os.path.join(output_dir, f'{prefix}_cut_edges.pdf')
    else:
        filename = os.path.join(output_dir, 'cut_edges.pdf')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def longest_boundary_fig(output_dir, prefix, normal, reversible):
    """Generates the normal vs. reversible ReCom longest boundary figure."""
    normal_longest_boundary = []
    reversible_longest_boundary = []
    for plan in normal:
        max_adj = np.max(plan.meta['district_adj'])
        normal_longest_boundary += [max_adj] * plan.multiplicity
    for plan in reversible:
        max_adj = np.max(plan.meta['district_adj'])
        reversible_longest_boundary += [max_adj] * plan.multiplicity
    plt.plot(normal_longest_boundary, label='ReCom')
    plt.plot(reversible_longest_boundary, label='Reversible ReCom')
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Longest Boundary')
    if prefix:
        filename = os.path.join(output_dir, f'{prefix}_longest_boundary.pdf')
    else:
        filename = os.path.join(output_dir, 'longest_boundary.pdf')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def demo_plans(output_dir, prefix, plans, n_plans, interval):
    steps_since_reset = 0
    interval_count = 0
    for plan in plans:
        steps_since_reset += plan.multiplicity
        while steps_since_reset > interval:
            fig, ax = plt.subplots()
            ax.axis('off')
            plan.partition.plot(ax=ax)
            idx = interval_count * interval
            plt.savefig(os.path.join(output_dir,
                                     prefix + f'_demo_{idx}.png'),
                        dpi=300,
                        bbox_inches='tight',
                        transparent=True)
            plt.close()
            steps_since_reset -= interval
            interval_count += 1


def acceptance_stats(output_dir, prefix, plans):
    statuses = defaultdict(int)
    total_steps = 0
    for step in plans:
        for reason, count in step.meta['reasons'].items():
            statuses[reason] += count
            total_steps += count
        statuses['accepted'] += 1
        total_steps += 1
    status_table = [
        {'status': status, 'count': count}
        for status, count in statuses.items()
    ]
    filename = os.path.join(output_dir, '_'.join([prefix, 'status.csv']))
    pd.DataFrame(status_table).to_csv(filename, index=False)


def election_hists(output_dir, prefix, updater, party, normal, reversible):
    n_plans = (sum(p.multiplicity for p in normal) +
               sum(p.multiplicity for p in reversible))
    n_districts = len(normal[0].partition)
    district_labels = np.zeros(n_plans * n_districts, dtype=int)
    district_shares = np.zeros(n_plans * n_districts)
    district_chain_labels = []
    idx = 0
    for plans, label in zip([normal, reversible],
                            ['ReCom', 'Reversible ReCom']):
        for plan in plans:
            shares = sorted(plan.partition[updater].percents(party))
            for district, share in enumerate(shares):
                district_labels[idx:idx + plan.multiplicity] = district + 1
                district_shares[idx:idx + plan.multiplicity] = share
                district_chain_labels += [label] * plan.multiplicity
                idx += plan.multiplicity

    sns.boxplot(y='share',
                x='district',
                data={
                    'share': district_shares,
                    'district': district_labels,
                    'chain': district_chain_labels
                },
                palette='colorblind',
                hue='chain')
    plt.ylim(0, 1)
    plt.xlabel('District')
    # TODO: don't hardcode human-readable party
    plt.ylabel('Party A vote share')
    plt.savefig(os.path.join(output_dir, prefix + '.pdf'),
                bbox_inches='tight')


def dump_run(output_dir, filename, plans):
    with open(os.path.join(output_dir, filename + '.json'), 'w') as f:
        run = []
        for plan in plans:
            run.append({
                'assignment': [plan.partition.assignment[node]
                               for node in sorted(plan.partition.assignment)],
                'multiplicity': plan.multiplicity,
                'district_adj': plan.meta['district_adj'].tolist(),
                'reasons': plan.meta['reasons']
            })
        json.dump(run, f)


@click.command()
@click.option('--graph-json')
@click.option('--shp')
@click.option('--n-steps', default=10000, type=int)
@click.option('--output-dir', required=True)
@click.option('--prefix', default='')
@click.option('--seed', default=0, type=int)
@click.option('--pop-col', default='population')
@click.option('--pop-tol', default=0.01, type=float)
@click.option('--plan-col', default='district')
@click.option('--reproject', is_flag=True)
@click.option('--election', nargs=2)
def main(graph_json, shp, n_steps, output_dir, prefix,
         seed, pop_col, pop_tol, plan_col, reproject, election):
    os.makedirs(output_dir, exist_ok=True)
    has_geometry = False
    if not shp and not graph_json:
        print('Specify a shapefile or a NetworkX-format graph '
              'JSON file.', file=sys.stderr)
        sys.exit(1)
    elif shp and not graph_json:
        gdf = gpd.read_file(shp)
        if reproject:
            gdf = reprojected(gdf)
        graph = Graph.from_geodataframe(gdf)
        has_geometry = True
    elif graph_json and not shp:
        graph = Graph.from_json(graph_json)
    else:
        graph = Graph.from_json(graph_json)
        gdf = gpd.read_file(shp)
        if reproject:
            gdf = reprojected(gdf)
        print('Appending geometries from shapefile to graph...')
        graph.geometry = gdf.geometry  # TODO: is this always valid?
        has_geometry = True

    my_updaters = {'population': updaters.Tally(pop_col, alias='population')}
    if election:
        election_up = Election('election', {'Democratic': election[0],
                                            'Republican': election[1]})
        my_updaters['election'] = election_up
    initial_state = GeographicPartition(graph,
                                        assignment=plan_col,
                                        updaters=my_updaters)

    normal_chain = RecomChain(graph=graph,
                              total_steps=n_steps,
                              initial_state=initial_state,
                              pop_col=pop_col,
                              pop_tol=pop_tol,
                              reversible=False,
                              seed=seed)
    reversible_chain = RecomChain(graph=graph,
                                  total_steps=n_steps,
                                  initial_state=initial_state,
                                  pop_col=pop_col,
                                  pop_tol=pop_tol,
                                  reversible=True,
                                  seed=seed)

    normal_plans = [plan for plan in tqdm(normal_chain)]
    reversible_plans = [plan for plan in tqdm(reversible_chain)]
    cut_edges_fig(output_dir, prefix, normal_plans, reversible_plans)
    longest_boundary_fig(output_dir, prefix, normal_plans, reversible_plans)
    if has_geometry:
        demo_plans(output_dir, '_'.join([prefix, 'recom']),
                   normal_plans, n_steps, n_steps // 25)
        demo_plans(output_dir, '_'.join([prefix, 'reversible_recom']),
                   reversible_plans, n_steps, n_steps // 25)
    if election:
        election_hists(output_dir, 'dem_vote_share',
                       'election', 'Democratic',
                       normal_plans, reversible_plans)
    acceptance_stats(output_dir, '_'.join([prefix, 'recom']),
                     normal_plans)
    acceptance_stats(output_dir, '_'.join([prefix, 'reversible_recom']),
                     reversible_plans)
    # dump_run(output_dir, '_'.join([prefix, 'recom']), normal_plans)
    # dump_run(output_dir, '_'.join([prefix, 'reversible_recom']),
    #          reversible_plans)


if __name__ == '__main__':
    main()
