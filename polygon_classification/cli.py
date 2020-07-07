"""Command line interface for operation management"""
import click
import os, sys

from polygon_classification.core import StrategyEvaluator
from polygon_classification.config import MLConf


@click.group(context_settings=dict(max_content_width=120, help_option_names=['-h', '--help']))
def cli():
    pass


@cli.command(help='tune various classifiers and select the best hyper-parameters on a train dataset')
@click.option('--dataset', default='train_dataset.csv', help='Name of train dataset')
@click.option('--classifiers', default='RandomForest', show_default=True, help='Comma separated classifiers to tune')
def train(dataset, classifiers):
    click.echo('Training algorithms...')

    options = {
        'dataset': os.path.join(os.getcwd(), 'data', dataset),
        'classifiers': classifiers.strip().split(',')
    }

    if not set(options['classifiers']).issubset(MLConf.classifiers):
        sys.exit(f'The accepted classifier names are: {",".join(MLConf.classifiers)}')

    if os.path.isfile(options['dataset']):
        StrategyEvaluator().train(**options)
    else:
        print("Train dataset file is not found!!!\n")


@cli.command(help='evaluate the effectiveness of the proposed methods')
@click.option('--dataset', default='test_dataset.csv', help='Name of test dataset')
def evaluate(dataset):
    click.echo('Running evaluation...')

    eval_data = os.path.join(os.getcwd(), 'data', dataset)
    if os.path.isfile(eval_data):
        StrategyEvaluator().exec_classifiers(eval_data)
    else:
        print("Test dataset file is not found!!!\n")


@cli.command(help='A complete process of distinct steps in figuring out the best ML algorithm with optimal '
                  'hyperparameters that best fits to data at hand for the polygon classification problem.')
# @click.option('--dataset', default='polypairs_dataset.csv', help='Folder of datasets')
@click.option('--train_dataset', default='train_dataset.csv', help='Name of train dataset')
@click.option('--test_dataset', default='test_dataset.csv', help='Name of test dataset')
@click.option('--classifiers', default='RandomForest', show_default=True,
              help='Comma separated classifiers to tune, train and evaluate')
def run(train_dataset, test_dataset, classifiers):
    click.echo('Running evaluation...')

    train_options = {
        'dataset': os.path.join(os.getcwd(), 'data', train_dataset),
        'classifiers': classifiers.strip().split(',')
    }
    test_options = {
        'dataset': os.path.join(os.getcwd(), 'data', test_dataset),
        'classifier': None
    }

    if not set(train_options['classifiers']).issubset(MLConf.classifiers):
        sys.exit(f'The accepted classifier names are: {",".join(MLConf.classifiers)}')

    if os.path.isfile(train_options['dataset']) and os.path.isfile(test_options['dataset']):
        # StrategyEvaluator().exec_classifiers(os.path.join(os.getcwd(), 'data', dataset))
        best_clf_name = StrategyEvaluator().train(**train_options)

        test_options['classifier'] = best_clf_name
        StrategyEvaluator().evaluate(**test_options)
    else:
        print("Input files in config are not found!!!\n")


cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(run)


if __name__ == '__main__':
    cli()
