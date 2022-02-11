import os
import glob

modules_content = """tensorcircuit
==================================================

.. toctree::
    :maxdepth: 2

    {}
"""

module_content = """tensorcircuit.{}
==================================================

.. automodule:: tensorcircuit.{}
    :members:
    :undoc-members:
    :show-inheritance:
"""

module_content2 = """tensorcircuit.{}
==================================================

"""

submodule_content = """tensorcircuit.{}
==================================================

.. automodule:: tensorcircuit.{}
    :members:
    :undoc-members:
    :show-inheritance:
"""

totree = """
.. toctree::
    
    {}
"""


def main():
    os.makedirs("./api", exist_ok=True)

    all_modules = []

    # Sub Module
    for subdir in glob.glob("../../tensorcircuit/*/"):

        if "_" in subdir:
            continue

        subdir = os.path.basename(os.path.normpath(subdir))
        if subdir == "applications":  # temporally disable this module
            continue

        all_submodules = []
        for module_name in glob.glob(f"../../tensorcircuit/{subdir}/*.py"):
            module_name = os.path.basename(module_name)[:-3]
            if module_name in [
                "__init__",
                "abstract_backend",
                "tf_ops",
                "jax_ops",
            ]:
                continue

            with open(f"./api/{subdir}_{module_name}.rst", "w") as f:
                f.write(
                    submodule_content.format(
                        f"{subdir}.{module_name}", f"{subdir}.{module_name}"
                    )
                )

            all_submodules.append(f"./{subdir}_{module_name}.rst")

        with open(f"./api/{subdir}.rst", "w") as f:
            f.write(
                module_content2.format(subdir)
                + totree.format("\n    ".join(sorted(all_submodules)))
            )

        all_modules.append(f"./api/{subdir}.rst")

    # Single file module
    for module_name in glob.glob("../../tensorcircuit/*.py"):

        module_name = os.path.basename(module_name)[:-3]
        if module_name == "__init__":
            continue

        with open(f"./api/{module_name}.rst", "w") as f:
            f.write(module_content.format(module_name, module_name, module_name))

        all_modules.append(f"./api/{module_name}.rst")

    with open("modules.rst", "w") as f:
        f.write(modules_content.format("\n    ".join(sorted(all_modules))))


if __name__ == "__main__":
    main()

# current weakness: cannot update outdated module names
