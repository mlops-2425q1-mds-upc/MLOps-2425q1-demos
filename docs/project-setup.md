# Project setup guide

This guide will help you set up your project structure and dependency manager. To define the project structure, we will use the [cookiecutter data science template](https://drivendata.github.io/cookiecutter-data-science/). This template is a good starting point for your project, as it provides a well-organized structure and best practices for data science projects.

We will use [Poetry](https://python-poetry.org) as the dependency manager for our project. Poetry is a Python packaging and dependency management tool that simplifies the process of managing dependencies and packaging your project.

## Prerequisites
- Python 3.8 or higher
- Pipx
- Cookiecutter Python package
- Poetry Python package
- Git
- GitHub account

## Steps
1. Install Pipx:
    ```bash
    python3 -m pip install --user pipx
    python3 -m pipx ensurepath
    ```

2. Install Cookiecutter and Poetry:
    ```bash
    pipx install cookiecutter-data-science
    pipx install poetry
    ```

3. Create a new project using the cookiecutter data science template:
    ```bash
    # From the parent directory where you want to create the project
    ccds
    ```

    Follow the prompts to create your project. You can choose the default options or customize them according to your needs. Since we will be using Poetry we will select `none` for the `environment_manager` option. In addition, Poetry stores the list of dependencies installed inside the `pyproject.toml` file. Hence, we will remove the `requirements.txt` file generated by cookicutter.

4. Change to the project directory:
    ```bash
    cd project-name
    ```
    Replace `project-name` with the name of your project.

5. Remove the `[build-system]` section from the `pyproject.toml` file:
    ```toml
    [build-system]
    requires = ["flit_core >=3.2,<4"]
    build-backend = "flit_core.buildapi"
    ```
    This section has to be removed before initializing Poetry.

6. Initialize a new Poetry project:
    ```bash
    poetry init
    ```

    Follow the prompts to update the `pyproject.toml` file. You can choose the default options or customize them according to your needs.

7. Add basic project dependencies to the `pyproject.toml` file:
    ```bash
    poetry add pandas numpy scikit-learn
    ```
    This will add the `pandas`, `numpy`, and `scikit-learn` packages as project dependencies.

8. Add basic development dependencies to the `pyproject.toml` file:
    ```bash
    poetry add -G dev black pylint pytest
    ```
    This will add the `black`, `pylint`, and `pytest` packages as development dependencies to the project.

9. Create a new repository on [GitHub](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository):

![Create a new repository](https://docs.github.com/assets/images/help/repository/repo-create.png)

10. Initialize a new Git repository in the project directory:
    ```bash
    git init
    ```
11. Add the project files to the Git repository:
    ```bash
    git add .
    ```
12. Commit the changes:
    ```bash
    git commit -m "Initial commit"
    ```
13. Add the remote repository URL:
    ```bash
    git remote add origin url-to-remote-repository
    ```
    Replace `url-to-remote-repository` with the URL of the remote repository you created in step 7.

14. Push the changes to the remote repository:
    ```bash
    git push -u origin branch-name
    ```
    This will push the changes to the `branch-name` branch of the remote repository.
    By default, newer versions of git set the initial default branch to `main`. Older versions use the branch name `master`.

Your project is now set up with the project structure and dependency manager. You can start working on your project by adding code, data, and other project-specific files to the project directory. Make sure to follow best practices for project organization and version control to ensure reproducibility and collaboration.
