import os
import glob
import shutil
from os.path import join as pj


class RSTGenerator:
    title_line = "=" * 80
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
        try:
            shutil.rmtree(self.dfolder)
        except FileNotFoundError:
            pass
        os.makedirs(self.dfolder)

    def write(self, path, content):
        if isinstance(content, list):
            content = "\n".join(content)

        with open(path, "w") as f:
            f.write(content.replace("\\", r"/"))

        print(f"Finish writing {path}")

    def _file_generate(self, package_parents):
        file_list = []
        for module_name in glob.glob(pj(self.pfolder, *package_parents, "*.py")):
            module_name = os.path.basename(module_name)[:-3]
            if module_name in self.ingnored_modules:
                continue

            rst_file = pj(self.dfolder, *package_parents, f"{module_name}.rst")
            name = f"{self.name}"
            for n in package_parents:
                name += f".{n}"
            name += f".{module_name}"
            content = [
                name,
                self.title_line,
                self.automodule.format(name),
            ]
            self.write(rst_file, content)
            if not package_parents:
                upper = self.dfolder
            else:
                upper = package_parents[-1]
            file_list.append(upper + f"/{module_name}.rst")
        for subdir in glob.glob(pj(self.pfolder, *package_parents, "*/")):
            if "_" in subdir:
                continue
            subdir = os.path.basename(os.path.normpath(subdir))
            os.makedirs(pj(self.dfolder, *package_parents, subdir), exist_ok=True)
            rst_file = pj(self.dfolder, *package_parents, f"{subdir}.rst")
            subdir_filelist = self._file_generate(package_parents + [subdir])

            name = f"{self.name}"
            for n in package_parents:
                name += f".{n}"
            name += f".{subdir}"
            content = [
                name,
                self.title_line,
                self.toctree.format("\n    ".join(sorted(subdir_filelist))),
            ]
            self.write(rst_file, content)

            if not package_parents:
                upper = self.dfolder
            else:
                upper = package_parents[-1]
            file_list.append(upper + f"/{subdir}.rst")
        return file_list

    def modules_file(self, file_list):
        """Write the modules.rst"""
        content = [
            self.name,
            self.title_line,
            self.toctree.format("\n    ".join(sorted(file_list))),
        ]
        self.write("modules.rst", content)

    def start(self):
        self.cleanup()
        file_list = self._file_generate([])
        self.modules_file(file_list)


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
