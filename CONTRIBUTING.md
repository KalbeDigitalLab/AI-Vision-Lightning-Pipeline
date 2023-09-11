# Contributing Standards

Here is a simple guideline to get you started with your first contribution.

1. Set up environment using `virtualenv` or PyTorch-CUDA Docker container from [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).
2. Use [issues](https://github.com/KalbeDigitalLab/AI-Vision-Lightning-Pipeline/issues) to discuss the suggested changes. Create an issue describing changes if necessary and add labels to ease orientation.
3. [Fork AI-Vision-Lightning-Pipeline](https://help.github.com/articles/fork-a-repo/) so you can make local changes and test them.
4. Create a new branch for the issue. The branch naming convention is enforced by the CI/CD so please make sure you are using `feature/***` or `hotfix/***` format otherwise it will fail.
5. Implement your changes along with relevant tests for the issue. Please make sure you are covering unit, integration and e2e tests where required.
6. Create a pull request against **main** branch.

## Requirements

Run `pip install -r requirements.txt` to install all dependencies needed to run this project. Run `pre-commit install` to setup `pre-commit` runners and enable pre-commit check before pushing changes to local or upstream git.

### Additional VSCode Extensions

- [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring), generate automated docstring format from codes or functions
- [Better Comments](https://marketplace.visualstudio.com/items?itemName=aaron-bond.better-comments), human friendly comment format by highliting important parts.
- [Python Indent](https://marketplace.visualstudio.com/items?itemName=KevinRose.vsc-python-indent), correcting python indentation.
- [Trainiling Spaces](https://marketplace.visualstudio.com/items?itemName=shardulm94.trailing-spaces), highlight trailing spaces in code.

## Formatting

We use the [autpep8](https://pypi.org/project/autopep8/) python linter. You can have your code auto-formatted by
running `pip install autopep8`, then `autopep8 --in-place --aggressive --aggressive <filename>` to automatically format a single file.

## Linting

We use the [flake8](https://pypi.org/project/flake8/) as the python linter. You can automatically format your code by running `pip install flake8`, then `flake8 path/to/dir/` to run `flake8` in that directory.

## Docstrings

We use NumPy Docstring Style. Please refer
to [this guideline](https://numpydoc.readthedocs.io/en/latest/format.html) to learn more about the styleguide of NumPy and how to implement it in this project.

## Testing

We use [pytest](https://docs.pytest.org/en/6.2.x/) for our tests. In order to make it easier, we also have a set of custom options defined in [conftest.py](conftest.py).

### Running Tests

#### Standard:

- `pytest .`: Run all tests with memory only.

### Extra Resources

If you feel lost with any of these sections, try reading up on some of these topics.

- Understand how to write [pytest](https://docs.pytest.org/en/6.2.x/) tests.
- Understand what a [pytest fixture](https://docs.pytest.org/en/6.2.x/fixture.html) is.
- Understand what [pytest parametrizations](https://docs.pytest.org/en/6.2.x/parametrize.html) are.
