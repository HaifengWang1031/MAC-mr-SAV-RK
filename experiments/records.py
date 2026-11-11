"""Compatibility imports for existing experiments and source-selected workers.

New callers use run_support.records and explicit run_support.source.provenance.
Old callers retain broad provenance discovery, now including run_support. This
module owns no second record implementation and never rewrites saved identities.
"""
from run_support.records import (
    PROJECT as PROJECT,
    begin_run as begin_run,
    digest as digest,
    fail_run as fail_run,
    finish_run as finish_run,
    load_record as load_record,
    reusable_run as reusable_run,
    run_identity as run_identity,
    save_result as save_result,
    save_scalar_corrections as save_scalar_corrections,
    verified as verified,
    write_json as write_json,
)
from run_support.source import legacy_provenance as provenance
