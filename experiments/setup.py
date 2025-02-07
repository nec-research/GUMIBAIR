import setuptools

setuptools.setup(
    name='mcbn_experiments',
    description="""
    A package with a collection of mcbn-based experiments.
    Includes training/testing in various settings & benchmarking against other models.
    """,
    version='0.0.1',
    packages=['mcbn_experiments'],
    install_requires=['mcbn'],
)