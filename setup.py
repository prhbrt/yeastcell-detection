# -*- coding: utf-8 -*-
from setuptools import setup


setup(
      name='yeastcelldetection',
      version=__import__('yeastcelldetection').__version__,

      description='Evaluation funcitons for yeast cell detection and tracking pipelines.',
      long_description='Evaluation funcitons for yeast cell detection and tracking pipelines.',
      
      url='https://github.com/prhbrt/yeastcells-detection/',

      author='Herbert Kruitbosch, Yasmin Mzayek',
      author_email='H.T.Kruitbosch@rug.nl, y.mzayek@rug.nl',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
      ],
      keywords='yeast cell detection, microscopy images, tif, tiff, image segmentation, tracking, computer vision',
      
      packages=['yeastcells'],
      install_requires=[
        'scikit-image>=0.17.2',
        'scikit-learn>=0.23.2,<0.24', # some threadpoolctl issues at 0.24
        'opencv-python>=4.4.0.46',
        'opencv-contrib-python>=4.4.0.46',
        'numpy>=1.19.1',
        'scipy>=1.5.2',
        'Shapely>=1.7.0'
        'tqdm>=4.51.0',
        'pandas>=1.1.4',
        'matplotlib>=3.1.1',
      ],
      zip_safe=True,
)
