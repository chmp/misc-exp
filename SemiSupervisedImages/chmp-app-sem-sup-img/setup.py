from setuptools import setup

setup(
    name='chmp-app-sem-sup-img',
    packages=["chmp.app.sem_sup_img"],
    package_dir={'': 'src'},
    install_requires=['PIL', 'numpy', 'tensorflow', 'click']
)
