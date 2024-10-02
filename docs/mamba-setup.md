# Commands
- `module load mamba/latest`
- for building user python environments, i.e., `mamba create -n myenv -c conda-forge`
- e.g. `mamba create -n myenv -c conda-forge python=3.11 pandas=2.2 pyevtk
- for source activating user python environments, i.e., `source activate myenv` only use "source" and no "conda" and even "mamba"
- to use jupyter in environments do `mkjupy <env_name>`
- To list available environments, run: `mamba info --envs`
- Adding dependencies/packages in existing environment i.e. 'mamba install -c <channel> <package>'
- `mamba install -c conda-forge scikit-learn`

# Resources
- Managing Python Modules Through the Mamba Environment Manager: https://asurc.atlassian.net/wiki/spaces/RC/pages/1905328428/Managing+Python+Modules+Through+the+Mamba+Environment+Manager
- 