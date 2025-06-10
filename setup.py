import ast
import glob
import os
import re
import shutil
import subprocess
import sys

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

cythonize_list = [
    "warp_ipc/ipc_integrator.py",
    "warp_ipc/collision_detection.py",
    "warp_ipc/sim_model.py",
]

ptx_arch_list = [
    "86",
    "80",
    "75",
    "61",
]

precompile_commands = [
    [sys.executable, "-m", "examples.grasp_soft_teddy", "--num_envs", "1", "--tactile"],
    [sys.executable, "-m", "examples.grasp_soft_teddy", "--num_envs", "1"],
    [sys.executable, "-m", "examples.peg", "--num_envs", "1"],
]

test_commands = [
    [sys.executable, "-m", "examples.grasp_soft_teddy", "--num_envs", "1", "--mesh", "teddy/teddy_f.vtk"],
    [sys.executable, "-m", "examples.grasp_soft_teddy", "--num_envs", "1", "--mesh", "teddy/teddy_f.vtk", "--tactile"],
    [sys.executable, "-m", "examples.peg", "--num_envs", "1"],
]

cleaned_folder_list = ["_warp_ci", ".mypy_cache", "build", "docs", "output", "scripts", ".vscode", "cache"]

# released_demos = [
#     "benchmark",
#     "grasp",
# ]


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
                error_msg = f"[ERROR] Unexpected Recompilation: {node.name}"
                node.body = [
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id="print", ctx=ast.Load()),
                            args=[ast.Constant(value=error_msg)],
                            keywords=[],
                        )
                    )
                ]
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
        for file in cythonize_list:
            fname = os.path.basename(file)
            fname_noext = os.path.splitext(fname)[0]
            # Optionally delete intermediate files
            try:
                os.remove(f"warp_ipc/{fname_noext}.c")
            except FileNotFoundError:
                pass

            subprocess.run(["stubgen", f"warp_ipc/{fname_noext}.py", "-o", "stubs"], check=True)

            shutil.move(
                f"stubs/warp_ipc/{fname_noext}.pyi",
                f"warp_ipc/{fname_noext}.pyi",
            )
            # Optionally delete intermediate files
            try:
                os.remove(f"warp_ipc/{fname_noext}.py")
            except FileNotFoundError:
                pass
        shutil.rmtree("stubs", ignore_errors=True)
    except Exception as e:
        print(f"Failed to generate .pyi: {e}")
        exit(1)


def ensure_warp_installed():
    import os
    import subprocess
    import sys

    warp_dir = os.path.abspath("./_warp_ci")
    commit = "e53a83b"

    if not os.path.exists(warp_dir):
        print("[INFO] Cloning warp from GitHub...")
        subprocess.run(["git", "clone", "https://github.com/YuyangLee/warp", warp_dir], check=True)

    os.chdir(warp_dir)
    print(f"[INFO] Checking out commit {commit}")
    subprocess.run(["git", "checkout", commit], check=True)

    print("[INFO] Building warp C++ libs...")
    subprocess.run([sys.executable, "build_lib.py"], check=True)

    print("[INFO] Installing warp...")
    subprocess.run([sys.executable, "-m", "pip", "install", "."], check=True)

    os.chdir("..")

    try:
        import warp

        print(f"[INFO] warp installed: {warp.__version__}")
    except ImportError:
        print("[ERROR] warp install failed!")
        sys.exit(1)


def precompile_ptx():
    # start from a cleaned
    for ptx_arch in ptx_arch_list:
        ptx_dir = f"./ptx/ptx_{ptx_arch}"
        if os.path.exists(ptx_dir):
            shutil.rmtree(ptx_dir)
        os.makedirs(ptx_dir)

        # precompile for different PTX archs
        for command in precompile_commands:
            print(command)
            subprocess.run(
                command,
                env={
                    "TACCEL_PTX_DIR": ptx_dir,
                    "TACCEL_PTX_ARCH": ptx_arch,
                },
                text=True,
                check=True,
            )

        # delete warp kernels that not for warp-ipc
        for item in os.listdir(ptx_dir):
            full_path = os.path.join(ptx_dir, item)
            if os.path.isdir(full_path) and "warp_ipc" not in item:
                shutil.rmtree(full_path)


def post_clean():
    for folder in cleaned_folder_list:
        if os.path.isdir(folder):
            shutil.rmtree(folder)

    # for demo in os.listdir("./examples"):
    #     full_path = os.path.join("./examples", demo)
    #     if demo not in released_demos and os.path.isdir(full_path):
    #         shutil.rmtree(full_path)


def post_test():
    for ptx_arch in ptx_arch_list:
        ptx_dir = f"./ptx/ptx_{ptx_arch}"
        for command in test_commands:
            print(f"[INFO] Running: {command}")
            result = subprocess.run(
                command,
                env={
                    "TACCEL_PTX_DIR": ptx_dir,
                    "TACCEL_PTX_ARCH": ptx_arch,
                },
                capture_output=True,
                text=True,
                check=False,
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            print(stdout)

            for keyword in ["ERROR", "WARNING", "Traceback (most recent call last)", "log_J is nan"]:
                if keyword in stdout or keyword in stderr:
                    for line in stdout.splitlines() + stderr.splitlines():
                        if keyword in line:
                            print(f"[ERROR LINE] {line}")
                    raise RuntimeError(f"[ERROR] Detected '{keyword}' in output of: {command}")

            lines = stdout.splitlines()
            match = None
            for line in reversed(lines):
                match = re.search(r"\[PNIter (\d+), MaxIter (\d+)\]", line)
                if match:
                    break

            if not match:
                raise RuntimeError(f"[ERROR] No matching '[PNIter A, MaxIter B]' line found in output of: {command}")

            A, B = int(match.group(1)), int(match.group(2))
            if B <= 5:
                raise RuntimeError(f"[ERROR] MaxIter {B} too small (<=5) in command: {command}")

            print(f"[PASS] {command} passed all checks.")

        # delete warp kernels that not for warp-ipc
        for item in os.listdir(ptx_dir):
            full_path = os.path.join(ptx_dir, item)
            if os.path.isdir(full_path) and "warp_ipc" not in item:
                shutil.rmtree(full_path)


# Custom build_ext command
# TODO: Compose into one .so
class CustomBuildExt(build_ext):
    def finalize_options(self):
        super().finalize_options()
        ensure_warp_installed()
        precompile_ptx()
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
            compiler_directives={"language_level": "3", "warn.undeclared": True},
        )
        for ext in self.extensions:
            ext._needs_stub = False

    def run(self):
        print("📄 Generating .pyi stubs...")
        super().run()
        generate_stubs()
        post_clean()
        post_test()


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
