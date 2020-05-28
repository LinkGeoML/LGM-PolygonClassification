"""Command line interface for operation management"""
import click
import os

from polygon_classification.core import StrategyEvaluator


@click.group(context_settings=dict(max_content_width=120, help_option_names=['-h', '--help']))
def cli():
    pass


@cli.command(help='tune various classifiers and select the best hyper-parameters on a train dataset')
@click.option('--dataset', default='train_dataset.csv', help='Name of train dataset')
@click.option('--classifiers', default='RandomForest', show_default=True, help='Comma separated classifiers to tune')
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


@cli.command(help='evaluate the effectiveness of the proposed methods')
@click.option('--dataset', default='test_dataset.csv', help='Name of test dataset')
@click.option('--classifier', default='RandomForest', show_default=True, help='ML classifier to predict')
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


@cli.command(help='A complete process of distinct steps in figuring out the best ML algorithm with optimal '
                  'hyperparameters that best fits to data at hand for the polygon classification problem.')
# @click.option('--dataset', default='polypairs_dataset.csv', help='Folder of datasets')
@click.option('--train_dataset', default='train_dataset.csv', help='Name of train dataset')
@click.option('--test_dataset', default='test_dataset.csv', help='Name of test dataset')
@click.option('--classifiers', default='RandomForest', show_default=True,
              help='Comma separated classifiers to tune, train and evaluate')
def run(train_dataset, test_dataset, classifiers):
    click.echo('Running evaluation')

    options = {
        'dataset': os.path.join(os.getcwd(), 'data', train_dataset),
        'classifiers': classifiers.strip().split(',')
    }
    options2 = {
        'dataset': os.path.join(os.getcwd(), 'data', test_dataset),
        'classifier': classifiers.strip().split(',')[0]
    }

    if os.path.isfile(os.path.join(os.getcwd(), 'data', train_dataset)) and \
            os.path.isfile(os.path.join(os.getcwd(), 'data', test_dataset)):
        # StrategyEvaluator().exec_classifiers(os.path.join(os.getcwd(), 'data', dataset))
        StrategyEvaluator().train(**options)
        StrategyEvaluator().evaluate(**options2)
    else:
        print("Input files in config are not found!!!\n")


cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(run)


if __name__ == '__main__':
    cli()
