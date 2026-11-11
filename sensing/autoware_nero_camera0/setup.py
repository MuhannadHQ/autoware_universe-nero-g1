from glob import glob
from setuptools import setup

package_name = "autoware_nero_camera0"
setup(
    name=package_name,
    version="0.44.2",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.xml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ahmed Alqurashi",
    maintainer_email="ahmed.q@usenero.com",
    description="camera0 (forward wide camera) rectification for perception",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["camera0_node = autoware_nero_camera0.camera0_node:main",
                                      "roi_class_filter = autoware_nero_camera0.roi_class_filter:main"]},
)
