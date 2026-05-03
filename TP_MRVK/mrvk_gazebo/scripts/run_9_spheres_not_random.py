#!/usr/bin/env python3
import importlib.util
import os
import sys

def load_impl():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    impl_path = os.path.join(script_dir, "bp_real_working", "run_9_spheres_not_random.py")
    if not os.path.exists(impl_path):
        raise FileNotFoundError("Missing implementation script: {}".format(impl_path))

    spec = importlib.util.spec_from_file_location("run_9_spheres_not_random_impl", impl_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def main():
    impl = load_impl()
    node = impl.NineSpheresDeterministicController()
    node.spin()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise
