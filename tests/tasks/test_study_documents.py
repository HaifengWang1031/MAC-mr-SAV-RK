"""Public documentation links and paper-setting contracts for experiment studies."""
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[2]


def _linked_materials(readme):
    return [(readme.parent/link.split('#')[0]).resolve()
            for link in re.findall(r'\]\(([^)]+)\)', readme.read_text())
            if '://' not in link and not link.startswith('#')]


def test_manufactured_study_has_real_notebook_and_embeddable_chinese_description():
    family = PROJECT/'experiments/manufactured_convergence'
    study = family/'studies/time_convergence'
    links = _linked_materials(family/'README.md')
    assert (study/'results.ipynb') in links
    assert (study/'description.tex') in links
    assert all(path.is_file() for path in links)
    text = (study/'description.tex').read_text()
    assert '\\documentclass' not in text and '\\begin{document}' not in text
    assert '半离散' in text and '空间收敛' in text
    assert re.search(r'\\nu\s*=\s*0\.01\b', text)
    assert re.search(r'\\gamma\s*=\s*1\b', text)
    assert re.search(r'T\s*=\s*2\b', text)
    assert '128' in text and '0.05' in text


def test_kolmogorov_study_links_only_the_description_and_explains_missing_data():
    family = PROJECT/'experiments/kolmogorov_adaptive'
    study = family/'studies/adaptive_fixed'
    links = _linked_materials(family/'README.md')
    assert (study/'description.tex') in links
    assert (study/'results.ipynb') not in links
    assert not (study/'results.ipynb').exists()
    assert 'Notebook 待补原始数据' in (family/'README.md').read_text()
    text = (study/'description.tex').read_text()
    assert '\\documentclass' not in text and '\\begin{document}' not in text
    assert re.search(r'\\nu\s*=\s*0\.02\b', text)
    assert re.search(r'\\gamma\s*=\s*1000\b', text)
    assert re.search(r'T\s*=\s*30\b', text)
    assert '256' in text and '自然' in text
    assert '局部' in text and '全局' in text
    notes = (study/'README.md').read_text()
    assert 'complete_with_unverified_reference' in notes
    assert '20261002T051000-1b29b1f9' in notes
    assert '20261004T002843-58006624' in notes


def test_results_navigation_links_real_materials_and_covers_existing_families():
    index = PROJECT/'experiments/README.md'
    links = _linked_materials(index)
    assert all(path.is_file() for path in links)
    assert NOTEBOOK_PATH in links
    assert PROJECT/'experiments/manufactured_convergence/studies/time_convergence/description.tex' in links
    assert PROJECT/'experiments/kolmogorov_adaptive/studies/adaptive_fixed/description.tex' in links
    assert '待补原始数据' in index.read_text()
    assert '材料状态' in index.read_text() and '数值验证' in index.read_text()
    # Discover families; this remains valid when another family is added.
    for family in (PROJECT/'experiments').iterdir():
        if family.is_dir() and (family/'README.md').is_file():
            assert family/'README.md' in links


def test_published_navigation_and_study_section_links_resolve():
    documents = [PROJECT/'experiments/README.md', PROJECT/'experiments/convergence-results.md',
                 *PROJECT.glob('experiments/*/studies/*/README.md')]
    for document in documents:
        for link in re.findall(r'\]\(([^)]+)\)', document.read_text()):
            if '://' in link or '#' not in link:
                continue
            path, anchor = link.split('#', 1)
            target = document.parent/path if path else document
            assert target.is_file(), (document, link)
            headings = {re.sub(r'[^\w -]', '', line.lstrip('# ').lower()).replace(' ', '-')
                        for line in target.read_text().splitlines() if line.startswith('#')}
            assert anchor in headings, (document, link)


def test_convergence_topic_separates_historical_forcing_and_unverified_references():
    topic = PROJECT/'experiments/convergence-results.md'
    assert all(path.is_file() for path in _linked_materials(topic))
    text = topic.read_text()
    assert '20260922T032418-3cd92bd4' in text and '20260922T054144-5aee3c91' in text
    assert 'f=(0,sin x)' in text and 'f=(0,sin 4*pi*x)' in text
    assert 'finer_reference_unverified' in text and '92' in text
    for study in ['forced_time_convergence', 'trig_time_convergence']:
        directory = PROJECT/'experiments/forced_ns_convergence/studies'/study
        assert directory/'results.ipynb' in _linked_materials(topic)
        assert directory/'description.tex' in _linked_materials(topic)
        assert all(path.is_file() for path in _linked_materials(directory/'README.md'))
        description = (directory/'description.tex').read_text()
        assert '\\documentclass' not in description and '\\begin{document}' not in description
        assert '严格' in description and ('空间' in description or '长期' in description)


NOTEBOOK_PATH = PROJECT/'experiments/manufactured_convergence/studies/time_convergence/results.ipynb'
