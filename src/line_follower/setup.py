from pathlib import Path
from setuptools import setup

# Resources live in the package so colcon and Docker produce the same install tree.
files = []
for root in ('launch', 'config', 'models', 'worlds'):
    for directory in sorted(Path(root).rglob('*')):
        if directory.is_file():
            files.append(('share/line_follower/' + str(directory.parent), [str(directory)]))
setup(name='line_follower', version='0.1.0', packages=['line_follower'],
      data_files=[('share/ament_index/resource_index/packages', ['resource/line_follower']),
                  ('share/line_follower', ['package.xml'])] + files,
      install_requires=['setuptools'], zip_safe=True,
      maintainer='Student', maintainer_email='student@example.com',
      description='Camera line follower', license='Apache-2.0',
      entry_points={'console_scripts': [
          'follower = line_follower.node:main',
          'reference_follower = line_follower.reference_node:main',
      ]})
