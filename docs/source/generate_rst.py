import os, glob

module_content = """tensorcircuit.{}
==========================================

.. toctree::
   :maxdepth: 4


tensorcircuit.{} module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: tensorcircuit.{}
    :members:
    :undoc-members:
    :show-inheritance:
"""

modules_content = """tensorcircuit
==========================================

.. toctree::
    :maxdepth: 4

    {}
"""
os.makedirs("./api", exist_ok=True)

all_modules = []
for module_name in glob.glob("../../tensorcircuit/*.py"):
    module_name = os.path.basename(module_name)[:-3]
    if module_name == "__init__":
        continue

    with open(f"./api/{module_name}.rst", "w") as f:
        f.write(module_content.format(module_name, module_name, module_name))
    all_modules.append(f"./api/{module_name}.rst")


for subdir in glob.glob("../../tensorcircuit/*/"):

    if "_" in subdir:
        continue

    subsir = os.path.basename(os.path.normpath(subdir))
    for module_name in glob.glob(f"../../tensorcircuit/{subdir}/*.py"):
        module_name = os.path.basename(module_name)[:-3]
        if module_name in [
            "__init__",
            "abstract_backend",
            "tf_ops",
            "jax_ops",
            "abstract_backend",
        ]:
            continue

        with open(f"./api/{subdir}_{module_name}.rst", "w") as f:
            f.write(
                module_content.format(
                    f"{subdir}.{module_name}",
                    f"{subdir}.{module_name}",
                    f"{subdir}.{module_name}",
                )
            )
        all_modules.append(f"./api/{subdir}_{module_name}.rst")

with open("modules.rst", "w") as f:
    f.write(modules_content.format("\n    ".join(all_modules)))
