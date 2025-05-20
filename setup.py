from setuptools import setup, find_packages

setup(
    name="mjx-safety-gym",
    version="0.1.0",
    description="MJX Safety Gym Environments",
    author="Max van der Hart",
    author_email="your.email@example.com",
    packages=find_packages(),  # Automatically finds all packages in the directory
    install_requires=[
        "jax==0.6.0",
        "flax==0.10.6",
        "mujoco-mjx==3.3.2",
        "pynput==1.8.1",
    ],
    python_requires=">=3.10",  # Adjust according to your project compatibility
)
