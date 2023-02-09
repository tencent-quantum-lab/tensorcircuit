import os
import glob
import shutil
from os.path import join as pj


class RSTGenerator:
    title_line = "=" * 50
    toctree = ".. toctree::\n    {}"
    automodule = ".. automodule:: {}\n    :members:\n    :undoc-members:\n    :show-inheritance:\n    :inherited-members:"

    def __init__(
        self, package_name, docs_folder, package_folder, ignore_modules=["__init__"]
    ):
        self.name = package_name
        self.dfolder = docs_folder
        self.pfolder = package_folder
        self.ingnored_modules = set(ignore_modules)
        self.tree = {}

    def cleanup(self):
        if os.path.exists("modules.rst"):
            os.remove("modules.rst")
        shutil.rmtree(self.dfolder)
        os.makedirs(self.dfolder)

    def write(self, path, content):
        if type(content) == type([]):
            content = "\n".join(content)

        with open(path, "w") as f:
            f.write(content.replace("\\", r"/"))

        print(f"Finish writing {path}")

    def single_file_module(self):
        """Process the module in the self.pfolder/*.py"""

        for module_name in glob.glob(pj(self.pfolder, "*.py")):
            module_name = os.path.basename(module_name)[:-3]
            if module_name in self.ingnored_modules:
                continue

            rst_file = pj(self.dfolder, f"{module_name}.rst")
            content = [
                f"{self.name}.{module_name}",
                self.title_line,
                self.automodule.format(f"{self.name}.{module_name}"),
            ]

            self.write(rst_file, content)
            self.tree[rst_file] = []

    def subdir_files_module(self):
        """Write the rst files for modules with subdir or files"""
        for subdir in glob.glob(pj(self.pfolder, "*/")):
            if "_" in subdir:
                continue

            subdir = os.path.basename(os.path.normpath(subdir))
            os.makedirs(pj(self.dfolder, subdir), exist_ok=True)
            rst_file = pj(self.dfolder, f"{subdir}.rst")
            self.tree[rst_file] = []

            for module_name in glob.glob(pj(self.pfolder, subdir, f"*.py")):
                module_name = os.path.basename(module_name)[:-3]
                if module_name in self.ingnored_modules:
                    continue

                content = [
                    f"{self.name}.{subdir}.{module_name}",
                    self.title_line,
                    self.automodule.format(f"{self.name}.{subdir}.{module_name}"),
                ]

                self.write(pj(self.dfolder, subdir, f"{module_name}.rst"), content)
                self.tree[rst_file].append(f"{subdir}/{module_name}.rst")

            content = [
                f"{self.name}.{subdir}",
                self.title_line,
                self.toctree.format("\n    ".join(sorted(self.tree[rst_file]))),
            ]
            self.write(rst_file, content)

    def modules_file(self):
        """Write the modules.rst"""
        content = [
            self.name,
            self.title_line,
            self.toctree.format("\n    ".join(sorted(self.tree.keys()))),
        ]
        self.write("modules.rst", content)

    def start(self):
        self.cleanup()
        self.single_file_module()
        self.subdir_files_module()
        self.modules_file()


if __name__ == "__main__":
    # All path must be relative path to the folder of moduels.rst
    RSTGenerator(
        "tensorcircuit",
        "./api",
        "../../tensorcircuit",
        [
            "__init__",
            "abstract_backend",
            "tf_ops",
            "jax_ops",
            "pytorch_ops",
            "asciiart",
        ],
    ).start()
