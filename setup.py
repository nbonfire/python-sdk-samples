from setuptools import setup

setup(name='python-sdk-sample',
      version='1.0',
      description='Sample for python SDK',
      url='https://bitbucket.org/Affectiva/python-sdk-samples/src/master/',
      author='Affectiva Engineering',
      author_email='affdexdev@affectiva.com',
      license='Proprietary',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering :: Image Recognition',

          # Pick your license as you wish (should match "license" above)
          'License :: Other/Proprietary License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 3.6',
      ],
      install_requires=[
          'opencv-contrib-python'
      ],
      scripts=[
          'python-sdk-sample/python-sample.py',
      ],

      )
