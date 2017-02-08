try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
      name = 'WtSubspace',
      packages = ['WtSubspace'], # this must be the same as the name above
      version = '0.1'
      description = 'Weighted random subspace method for classification',
      author = 'Peng Wu',
      license='LICENSE.txt',
      author_email = 'pwusff@gmail.com',
      long_description=open('README.md').read(),
      install_requires=["scikit-learn >= 0.17", "pandas >= 0.17.1"]
      )