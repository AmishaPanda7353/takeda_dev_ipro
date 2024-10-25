import io
import os
import os.path as op

from setuptools import find_packages, setup


def get_install_req():
    """
    Reads the requirements from the appropriate requirements file based on the operating system.

    Returns:
        list: A list of strings containing the required Python packages.
    """
    if os.name == "nt":
        with io.open("deploy/requirements_windows.txt") as fh:
            install_reqs = fh.read()
    else:
        with io.open("deploy/requirements.txt") as fh:
            install_reqs = fh.read()
    # Split the requirements into lines and filter out any empty lines
    install_reqs = [l for l in install_reqs.split("\n") if len(l) > 1]
    return install_reqs


def get_version_info():
    """
    Extract version information as a dictionary from version.py
    """
    version_info = {}
    version_filename = os.path.join("src", "query_insights", "utils", "version.py")
    with open(version_filename, "r") as version_module:
        version_code = compile(version_module.read(), "version.py", "exec")
    exec(version_code, version_info)
    return version_info

# read the readme file for accessing the project descriptions/details
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# metadata for setup functions
metadata = dict(
    name="insights_pro",
    version=get_version_info()["version"],
    description="Tiger Capability Solution",
    long_description=long_description,
    author="Tiger Analytics",
    download_url="https://github.com/tigerrepository/Insights-Pro",
    project_urls={
        "Bug Tracker": "https://github.com/tigerrepository/Insights-Pro/issues",
        "Source Code": "https://github.com/tigerrepository/Insights-Pro",
    },
    platforms=["Windows", "Linux", "Unix"],
    test_suite="unittest",
    python_requires=">=3.8.1",
    zip_safe=False,
    install_requires=get_install_req(),
    packages=find_packages(),
    include_package_data=True,
)

# calling the setup function
setup(**metadata)
