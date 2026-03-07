to run all tests from the root of the repository:
```bash
python -m unittest discover -s tests -v
```

tests are now grouped into categories:
- `tests/regression`: output/baseline regression tests
- `tests/nutil_comparison`: behavior and mapping-comparison tests
- `tests/core`: core invariants and validation tests

to run a single category:
```bash
python -m unittest discover -s tests/regression -t tests -v
python -m unittest discover -s tests/nutil_comparison -t tests -v
python -m unittest discover -s tests/core -t tests -v
```