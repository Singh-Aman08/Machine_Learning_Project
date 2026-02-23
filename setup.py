from setuptools import find_packages, setup

def requirement(file_path):
    with open(file_path) as file:
        requirement = file.readlines()
        requirement = [_.strip() for _ in  requirement ]
        return requirement

setup(
   name='ml_project',
   version='0.1',
   author='Aman Kumar Singh',
   author_email='amsrap7045@gmail.com',
   packages=find_packages(),
   install_requires=requirement("requirements.txt")
)