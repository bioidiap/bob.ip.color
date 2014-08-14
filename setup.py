#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 30 Jan 08:45:49 2014 CET

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.blitz', 'bob.core', 'bob.io.base']))
from bob.blitz.extension import Extension, Library, build_ext

import os
package_dir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.join(package_dir, 'bob', 'ip', 'color')

version = '2.0.0a0'

setup(

    name='bob.ip.color',
    version=version,
    description='Color conversion utilities',
    url='http://github.com/bioidiap/bob.ip.color',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    install_requires=[
      'setuptools',
      'bob.blitz',
      'bob.core',
      'bob.io.base',
    ],

    namespace_packages=[
      "bob",
      "bob.ip",
    ],

    ext_modules = [
      Extension("bob.ip.color.version",
        [
          "bob/ip/color/version.cpp",
        ],
        bob_packages = ['bob.core', 'bob.io.base'],
        version = version,
      ),

      Library("bob_ip_color",
        [
          "bob/ip/color/cpp/color.cpp",
        ],
        version = version,
        package_directory = package_dir,
        target_directory = target_dir,
        bob_packages = ['bob.core', 'bob.io.base'],
      ),

      Extension("bob.ip.color._library",
        [
          "bob/ip/color/utils.cpp",
          "bob/ip/color/rgb_to_gray.cpp",
          "bob/ip/color/rgb_to_yuv.cpp",
          "bob/ip/color/rgb_to_hsv.cpp",
          "bob/ip/color/rgb_to_hsl.cpp",
          "bob/ip/color/main.cpp",
        ],
        version = version,
        bob_packages = ['bob.core', 'bob.io.base'],
        libraries = ['bob_ip_color']
      ),
    ],

    cmdclass = {
      'build_ext': build_ext
    },

    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
    ],

  )
