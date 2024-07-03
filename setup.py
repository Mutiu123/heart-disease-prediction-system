from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
REPO_NAME = "Mutiu Adegboye"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = ["streamlit"]

setup(
    name = SRC_REPO,
    version = "0.0.1",
    author = REPO_NAME,
    author_email = "adegboyemutiu@gmail.com",
    description = "heart disease prediction system",
    long_description = long_description,
    long_description_content_type = "text/marckdown",
    packages = [SRC_REPO],
    python_requires = ">=3.10.14",
    install_requires = LIST_OF_REQUIREMENTS
    
    
)