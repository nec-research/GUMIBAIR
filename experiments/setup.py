import setuptools

setuptools.setup(
    name='gumibair_experiments',
    description="""
    A package with a collection of gumibair-based experiments.
    Includes training/testing in various settings & benchmarking against other models.
    """,
    version='0.0.1',
    packages=['gumibair_experiments'],
    install_requires=['gumibair'],
)