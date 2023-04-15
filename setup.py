from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function returns a list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Predicts whether two questions or sentences are similar to each other. Helps in deduplication of questions asked in forums like Quora, Reddit etc. Uses NLP techniques to build features.',
    author='Parikshit',
    license='MIT',
    install_requires=get_requirements('requirements.txt')    
)
