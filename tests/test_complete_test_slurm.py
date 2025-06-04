import os
from pathlib import Path


def test_complete_test_slurm_exists():
    root = Path(__file__).resolve().parents[1]
    script = root / 'complete_test.slurm'
    assert script.is_file()

    text = script.read_text()
    assert 'generate_clean_configs' in text
    assert 'test_both_plumes_complete' in text
    assert '#SBATCH --output=slurm_logs/complete_test/complete_test_logs_%j.out' in text


def test_complete_test_slurm_logging():
    """Check that the script logs diagnostics with a timestamp."""
    root = Path(__file__).resolve().parents[1]
    text = (root / 'complete_test.slurm').read_text()
    assert 'TIMESTAMP=$(date +%Y%m%d_%H%M%S)' in text
    assert 'diagnostics_${TIMESTAMP}' in text
