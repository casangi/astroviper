"""AGENTS.md section 6 rule 6, checked on the source without building it: a C++
kernel given a thread count below 1 runs one thread, never
std::thread::hardware_concurrency(). Only the vendored pocketfft, whose own
count of 0 means the hardware concurrency, names it."""

from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / "src/astroviper/processing_functions"
#: The vendored files that may name hardware_concurrency, relative to SOURCE.
VENDORED = ["include/pocketfft_hdronly.h"]


def test_no_kernel_source_names_the_hardware_concurrency():
    sources = sorted(
        path.relative_to(SOURCE).as_posix()
        for path in SOURCE.rglob("*")
        if path.suffix in {".cpp", ".hpp", ".h"}
    )
    assert set(VENDORED) <= set(sources), SOURCE
    named = [
        source
        for source in sources
        if source not in VENDORED
        and b"hardware_concurrency" in (SOURCE / source).read_bytes()
    ]
    assert not named, "hardware_concurrency in:\n" + "\n".join(named)
