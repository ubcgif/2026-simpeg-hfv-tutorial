# Time-domain hierarchical finite volume tutorial for AQGEO 2026 Meeting

## Information

|                            | **Information**                                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------- |
| When                       | November 6, 2026                                                                                          |
| Maxwell plate notebook     | [notebooks/fwd_utem_hierarchical_3d_part_1.ipynb](notebooks/fwd_utem_hierarchical_3d_part_1.ipynb)        |
| Delaunay surface notebook  | [notebooks/fwd_utem_hierarchical_3d_part_2.ipynb](notebooks/fwd_utem_hierarchical_3d_part_1-sparse.ipynb) |

## How to run the notebooks

### Install a Python distribution

In order to run the notebooks you need to install a Python distribution like
[Miniforge][miniforge], [Anaconda][anaconda] or [Miniconda][miniconda].

Follow the instructions provided by any of them to install it in your system.

> [!NOTE]
> In the following steps we'll instruct you to run some commands. They are
> meant to be run in a terminal (for Mac and Linux users), or in an Anaconda
> Prompt or Miniforge Prompt if you are running Windows.

### Download the repository

Then, download this repository as a [zip file][repo-zip], or clone it with `git`:

```bash
git clone https://github.com/ubcgif/2026-hierarchical-utem-tutorial.git
```

### Create a new conda environment

In the downloaded folder you'll find an `environment.yml` file that you can use
to create a conda environment by running:

```bash
conda env create -f environment.yml
```

Once the new environment is created and all the packages are installed, we need
to activate the environment with:

```bash
conda activate simpeg-hfv-tutorial
```

## Run JupyterLab

After we activated the environment, let's start JupyterLab, where we can open,
edit and execute the notebooks:

```bash
jupyter lab
```

It will automatically open a new tab in your browser with the Jupyter Lab, where
you can navigate into the `notebooks` folder and open any of the notebooks
you'll find there.

> [!IMPORTANT]
> It's important not to close the terminal (or Prompt) window while you are
> using Jupyter Lab. Closing the terminal will make Jupyter Lab to quit.

## License

The slides and their content are provided under the [CC-BY License](LICENSE).

The code under the `notebooks` folder is provided under the [MIT
License](notebooks/LICENSE).

[anaconda]: https://anaconda.org
[miniconda]: https://docs.anaconda.com/miniconda/miniconda-install
[miniforge]: https://github.com/conda-forge/miniforge
[repo-zip]: https://github.com/ubcgif/2026-hierarchical-utem-tutorial/archive/refs/heads/main.zip
