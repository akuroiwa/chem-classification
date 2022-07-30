# -*- coding: utf-8 -*-

import glob
from setuptools import setup, find_packages

setup(
    name='chem_classification',
    version='0.0.1',
    url='https://github.com/akuroiwa/chem-classification',
    # # PyPI url
    # download_url='',
    license='GNU/GPLv3+',
    author='Akihiro Kuroiwa',
    author_email='akuroiwa@env-reform.com',
    description='Deep learning in smiles win / loss evaluation.',
    # long_description="\n%s" % open('README.md').read(),
    long_description=open("README.md", "r").read(),
    long_description_content_type='text/markdown',
    zip_safe=False,
    # python_requires=">=3.8",
    python_requires=">=3.7",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        "Operating System :: OS Independent",
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3 :: Only',
        # 'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    platforms='any',
    keywords=['classification', 'transformer', 'roberta', 'cheminformatics', 'chemoinformatics'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=['simpletransformers', 'pandas'],
    entry_points={
        'console_scripts': [
            'importSmiles = chem_classification.importSmiles:console_script'
            ]},
)
