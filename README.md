# A simple ANN Wrapper

This framework implements artificial neural networks (ANN) together with some common datasets. It can be used as a package, where you can easily instantiate complex networks and (re-)train them or just run already trained once.

There is a [main](src/main.py) script, that helps getting used to the framework. Please run

```
python3 main.py -h
```

within the [src](src) folder to get an overview of this script.

In short, the [main](src/main.py) script will ensure that
 * the output folder is corretly initialized,
 * the logger is set up,
 * the desired random seed is enforced.
 
The default values of all command-line parameters are specified in the module [configuration](src/configuration.py).

## Example Scripts

There are several example scripts in the folder [src/examples](src/examples), that can be used in combination with the [main](src/main.py) script.

Here is an example of how to train an MNIST autoencoder:

```
python3 main.py -r examples.mnist_autoencoder
```

Or, if running the program on a machine without a graphical interface:

```
python3 main.py -r examples.mnist_autoencoder -k "{\"allow_plots\": true}"
```

### Implementing new Example Scripts

One can easily write custom example scripts. The only requirement is the existence of a method `run(**kwargs)`. Please checkout existing scripts.

## Implemented Datasets

The following datasets are currently supported by the framework.

* MNIST
* CIFAR-10
