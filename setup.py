from setuptools import setup,find_packages
from typing import List

HYPHON_E_DOT = '-e .'
def get_requirements(file_path:str)->List:
    with open(file_path) as fp:
        requirements = fp.readlines()
        reqList = [req.replace('\n','') for req in requirements]
        if HYPHON_E_DOT in reqList:
            reqList.remove(HYPHON_E_DOT)

    return reqList



setup(
    name="ProjectName",
    version='0.0.1',
    author="Durgesh Chandra Mishra",
    author_email="durgeshcmishra@gmail.com",
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)