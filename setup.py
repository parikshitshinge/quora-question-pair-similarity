from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Predicts whether two questions or sentences are similar to each other. Helps in deduplication of questions asked in forums like Quora, Reddit etc. Uses NLP techniques to build features.',
    author='Parikshit',
    license='MIT',
)
