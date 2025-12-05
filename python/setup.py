from setuptools import setup, find_packages

package_name = 'kalman_positioning_python'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mohammad Sherif',
    maintainer_email='your_email@example.com',
    description='Python UKF implementation for robot localization',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'positioning_node = ros2_positioning_node:main',
        ],
    },
)

