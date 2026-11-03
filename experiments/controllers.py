"""Shared experiment controller construction and finite JSON settings.

Experiments own their default tolerances and scalar-control eligibility. This
module owns only the common I/PI controls and embedded-scheme order contract.
"""
from dataclasses import asdict
from typing import Any

import numpy as np

from solver.adaptivity import IController, PIController
from solver.schemes.registry import scheme_info


def make_controller(config: dict[str, Any], name: str) -> IController:
    options = dict(config)
    kind = options.pop('kind', 'PI')
    if kind not in ('I', 'PI'):
        raise ValueError('Controller kind must be I or PI')
    order = scheme_info(name).estimator_order
    if order is None:
        raise ValueError('IMEX Euler requires a separate step-doubling estimator; use fixed mode')
    if options.get('estimator_order', order) != order:
        raise ValueError('Estimator order must match the embedded scheme')
    options['estimator_order'] = order
    if options.get('max_step', 1.) is None:
        options['max_step'] = float('inf')
    for key in ('max_steps', 'max_rejections'):
        if key in options and type(options[key]) is not int:
            raise ValueError(f'{key} must be an integer')
    try:
        return (IController if kind == 'I' else PIController)(**options)
    except TypeError as error:
        raise ValueError(f'Invalid controller configuration: {error}') from error


def controller_settings(controller: IController) -> dict[str, Any]:
    """Preserve canonical settings, representing an unbounded maximum as null."""
    values = asdict(controller)
    values['max_step'] = None if np.isinf(controller.max_step) else controller.max_step
    return {'kind': 'PI' if isinstance(controller, PIController) else 'I', **values}
