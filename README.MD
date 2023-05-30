# Hydro-ML

Hydro-ML is an application designed for hydrological forecasting using machine learning models.

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

### Creating a Kernel

To create a Python kernel named `hydro-ml` which can be used with tools like VSCode, Jupyter notebooks, etc., use the following command:

```bash
poetry run python -m ipykernel install --user --name=hydro-ml
```
The above command will install a Python kernel named hydro-ml which can be used to run this project's code in interactive environments like notebooks.

### Running the Application
The Hydro-ML application is a web application built using the Streamlit framework. You can run the application with the following command:
```bash
poetry run streamlit run app/app.py
```
This command will start the Hydro-ML application. You should be able to access the application by opening your web browser and visiting 'http://localhost:8501'.
