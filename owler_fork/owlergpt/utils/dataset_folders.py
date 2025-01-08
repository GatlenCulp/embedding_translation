""" "
`utils.choose_dataset_folders` module: Choose a dataset folder from a list of available ones.
"""

import os

import click


def choose_dataset_folders(dataset_path: str) -> list[str] | None:
    """Fetches and lists all available dataset folders to allow the user to choose for which dataset embeddings should be
    created or evaluated.

    :param dataset_path: The folder in which datasets are located.
    :return: The selected dataset folder.
    """
    dataset_folders = [
        f
        for f in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, f))
    ]
    if not dataset_folders:
        click.echo(f"No dataset folders found. Dataset path was {dataset_path}")
        return None

    click.echo("Available dataset folders:")
    for idx, folder_name in enumerate(dataset_folders, start=1):
        click.echo(f"{idx}. {folder_name}")

    prompt = 'Please enter the number of the folder you want to process (comma-seperated or "all" for all)'

    folder_choice = click.prompt(prompt, type=str, default="1")
    if folder_choice == "all":
        return dataset_folders

    folder_choices = [int(folder) - 1 for folder in folder_choice.split(",")]
    if any(folder < 0 or folder >= len(dataset_folders) for folder in folder_choices):
        click.echo("Invalid selection. Exiting.")
        return None

    selected_folders = [dataset_folders[folder] for folder in folder_choices]
    click.echo(f"Selected folders: {selected_folders}")
    return selected_folders
