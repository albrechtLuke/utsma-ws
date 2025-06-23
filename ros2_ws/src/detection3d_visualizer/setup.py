from setuptools import setup
import os
from glob import glob

package_name = 'detection3d_visualizer'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',  
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resource'), glob('resource/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='luke_albrecht',
    maintainer_email='luke.albrecht@student.uts.edu.au',
    description='3D detection to MarkerArray for RViz2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection3d_to_markers = detection3d_visualizer.detection3d_to_markers:main',
        ],
    },
)

