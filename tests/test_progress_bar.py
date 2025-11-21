import io
from contextlib import redirect_stdout

from modules.ProgressBar import ProgressBar


def test_progress_bar_reaches_total_and_prints_label():
    buf = io.StringIO()
    bar = ProgressBar(total=5, label="Test", width=10)

    with redirect_stdout(buf):
        for _ in range(5):
            bar.update()
        bar.finish()

    output = buf.getvalue()
    assert "Test" in output
    assert "5/5" in output
