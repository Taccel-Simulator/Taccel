import glob
import shutil
import subprocess
import os
import ast
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

cythonize_list = [
    "warp_ipc/ipc_integrator.py",
    "warp_ipc/collision_detection.py",
    "warp_ipc/sim_model.py",
]


def clean_wp_file(file_path):
    with open(file_path, "r") as file:
        file_content = file.read()
    tree = ast.parse(file_content)
    modified = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if any(
                isinstance(decorator, ast.Attribute)
                and decorator.attr in ["kernel", "func"]
                and isinstance(decorator.value, ast.Name)
                and decorator.value.id == "wp"
                for decorator in node.decorator_list
            ):
                node.body = [ast.Pass()]
                modified = True
    modified_code = ast.unparse(tree)
    with open(file_path, "w") as file:
        file.write(modified_code)
    if modified:
        print(f"Modified {file_path}")
    return modified


def clean_wp(file_list):
    no_modify_path = []
    for path in file_list:
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith(".py") and os.path.join(path, file) not in cythonize_list:
                    file_path = os.path.join(path, file)
                    no_wp = not clean_wp_file(file_path)
        else:
            no_wp = not clean_wp_file(path)
        if no_wp:
            no_modify_path.append(path)
    return no_modify_path


def generate_stubs():
    try:
        subprocess.run(["stubgen", "-p", "warp_ipc", "-o", "stubs"], check=True)
        for file in cythonize_list:
            fname = os.path.basename(file)
            fname_noext = os.path.splitext(fname)[0]
            shutil.move(
                f"stubs/warp_ipc/{fname_noext}.pyi",
                f"warp_ipc/{fname_noext}.pyi",
            )
            # Optionally delete intermediate files
            try:
                os.remove(f"warp_ipc/{fname_noext}.py")
                os.remove(f"warp_ipc/{fname_noext}.c")
            except FileNotFoundError:
                pass
        shutil.rmtree("stubs", ignore_errors=True)
    except Exception as e:
        print(f"Failed to generate .pyi: {e}")


# Custom build_ext command
# TODO: Compose into one .so
class CustomBuildExt(build_ext):
    def finalize_options(self):
        super().finalize_options()
        print("🔧 Cleaning warp_ipc decorators before Cythonizing...")
        clean_wp(glob.glob("warp_ipc/**/*.py", recursive=True))
        print("⚙️ Running Cython...")
        self.extensions = cythonize(
            [
                Extension(
                    name=f.replace("/", ".").replace(".py", ""),
                    sources=[f],
                )
                for f in cythonize_list
            ],
            compiler_directives={"language_level": "3"},
        )
        for ext in self.extensions:
            ext._needs_stub = False

    def run(self):
        print("📄 Generating .pyi stubs...")
        super().run()
        generate_stubs()


setup(
    name="warp_ipc",
    version="1.0.0",
    packages=["warp_ipc"],
    zip_safe=False,
    ext_modules=[],
    package_data={"warp_ipc": ["*.pyi", "*.so", "*.py"]},
    exclude_package_data={"warp_ipc": [os.path.basename(f) for f in cythonize_list]},
    cmdclass={"build_ext": CustomBuildExt},
)
