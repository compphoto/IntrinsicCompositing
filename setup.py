import setuptools
setuptools.setup(
    name="intrinsic_compositing",
    version="0.0.1",
    author="Chris Careaga",
    author_email="chris_careaga@sfu.ca",
    description='a package containing to the code for the paper "Intrinsic Harmonization for Illumination-Aware Compositing"',
    url="",
    packages=setuptools.find_packages(),
    license="",
    python_requires=">3.6",
    install_requires=[
        'altered_midas @ git+https://github.com/CCareaga/MiDaS@master',
        'chrislib @ git+https://github.com/CCareaga/chrislib@main',
        'omnidata_tools @ git+https://github.com/CCareaga/omnidata@main',
        'boosted_depth @ git+https://github.com/CCareaga/BoostingMonocularDepth@main',
        'intrinsic @ git+https://github.com/compphoto/intrinsic@d9741e99b2997e679c4055e7e1f773498b791288'
    ]
)
