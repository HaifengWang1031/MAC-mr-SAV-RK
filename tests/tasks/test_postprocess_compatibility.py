"""Historical research/tool imports remain adapters to the public readonly owners."""


def test_old_plotting_analysis_and_snapshot_names_are_the_same_implementation() -> None:
    from experiments import analysis, plotting
    from tools import plot_speed_snapshots
    from postprocess import diagnostics, mac, plotting as figures, snapshots

    assert analysis.analyze_runs is diagnostics.analyze_runs
    assert plotting.read_run_fields is mac.read_run_fields
    assert plot_speed_snapshots.report is snapshots.report
    assert plot_speed_snapshots.main is snapshots.main
    for name in ('field_figure', 'history_figure', 'latex_binary', 'latex_style', 'usetex_available'):
        assert getattr(plotting, name) is getattr(figures, name)
    # Hard-coded research convergence policy is intentionally not a core figure.
    assert plotting.reference_error_figure.__module__ == 'experiments.plotting'
