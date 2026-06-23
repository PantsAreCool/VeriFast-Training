from pathlib import Path
import importlib.metadata

def check_project_dependencies(requirements_filename="requirements.txt"):
    req_path = Path(requirements_filename)
    
    if not req_path.exists():
        print(f"Error: '{requirements_filename}' could not be found.")
        return

    missing_packages = 0
    version_mismatches = 0
    matching_packages = 0

    with open(req_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
                
            if "==" in line:
                package_name, expected_version = line.split("==", 1)
                package_name = package_name.strip()
                expected_version = expected_version.strip()
            else:
                package_name = line.strip()
                expected_version = None

            try:
                installed_version = importlib.metadata.version(package_name)
                
                if expected_version and installed_version != expected_version:
                    print(f"[MISMATCH] {package_name}: Expected {expected_version}, but found {installed_version} installed.")
                    version_mismatches += 1
                else:
                    print(f"[OK] {package_name} ({installed_version}) is active.")
                    matching_packages += 1
                    
            except importlib.metadata.PackageNotFoundError:
                print(f"[MISSING] {package_name} is completely absent from the environment.")
                missing_packages += 1

    print("\nSummary:")
    print(f"Fully Intact: {matching_packages}")
    print(f"Version Mismatches: {version_mismatches}")
    print(f"Missing From Environment: {missing_packages}")

if __name__ == "__main__":
    test_reqs = Path("requirements.txt")
    test_reqs.write_text("openai==1.30.0\nfastapi\nnon_existent_package_xyz==2.1")

    check_project_dependencies("requirements.txt")

    test_reqs.unlink()