"""Command line interface for operation management"""
import click
import os

from polygon_classification.core import StrategyEvaluator


@click.group()
def cli():
    pass


@cli.command()
@click.option('--dataset', default='train_dataset.csv', help='Path to train dataset')
@click.option('--classifiers', default='RandomForest', help='ML classifiers to tune')
def train(dataset, classifiers):
    click.echo('Training algorithms')

    options = {
        'dataset': os.path.join(os.getcwd(), 'data', dataset),
        'classifiers': classifiers.strip().split(',')
    }

    if os.path.isfile(options['dataset']):
        StrategyEvaluator().train(**options)
    else:
        print("Train dataset file is not found!!!\n")


@cli.command()
@click.option('--dataset', default='test_dataset.csv', help='Path to test dataset')
@click.option('--classifier', default='RandomForest', help='ML classifier to predict')
def evaluate(dataset, classifier):
    click.echo('Running evaluation')

    options = {
        'dataset': os.path.join(os.getcwd(), 'data', dataset),
        'classifier': classifier
    }

    if os.path.isfile(options['dataset']):
        StrategyEvaluator().evaluate(**options)
    else:
        print("Test dataset file is not found!!!\n")


@cli.command()
@click.option('--dataset', default='polypairs_dataset.csv', help='Folder of datasets')
def run(dataset):
    click.echo('Running evaluation')

    if os.path.isfile(os.path.join(os.getcwd(), 'data', dataset)):
        StrategyEvaluator().exec_classifiers(os.path.join(os.getcwd(), 'data', dataset))
    else:
        print("Input files in config are not found!!!\n")


cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(run)


if __name__ == '__main__':
    cli()
