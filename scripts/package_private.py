"""Assemble or seal the private release kit. Never upload it to public Git."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import tarfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def verify_tree(root, manifest, expected_sha):
    root = Path(root)
    if sha(manifest) != expected_sha:
        raise ValueError("unapproved_study_manifest")
    records = json.loads(Path(manifest).read_text())
    for name, expected in records.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid_manifest_path")
        path = root / relative
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError("study_integrity_check_failed")
    return len(records)


def assemble(args):
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("invalid_release_commit")
    spec = json.loads(args.selection.read_text())
    verify_tree(args.study, args.study / "freeze-manifest.json", spec["study_freeze_sha256"])
    verify_tree(args.challenger, args.challenger / "freeze-manifest.json", spec["archived_challenger_freeze_sha256"])
    for root in (args.study, args.challenger):
        for p in root.rglob("*"):
            if p.is_symlink() and not (root == args.challenger and p == root / "data"
                    and p.resolve() == (args.study / "data").resolve()):
                raise ValueError("unexpected_private_study_link")
    for name, path in (("toxic-chat_annotation_test.csv", args.toxic_test), ("xstest_prompts.csv", args.xstest)):
        if sha(path) != spec["external_regression_files"][name]["sha256"]:
            raise ValueError("external_source_hash_mismatch")
    root = args.output
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    code = root / "code"
    code.mkdir(mode=0o700)
    with tarfile.open(args.source_archive) as archive:
        for member in archive.getmembers():
            p = Path(member.name)
            if p.is_absolute() or ".." in p.parts or not (member.isfile() or member.isdir()):
                raise ValueError("invalid_release_source_archive")
        archive.extractall(code, filter="data")
    if sha(code / "src/release/selected.json") != sha(args.selection):
        raise ValueError("archive_selection_mismatch")
    (root / "studies").mkdir(mode=0o700)
    shutil.copytree(args.study, root / "studies" / spec["selected_study_id"], symlinks=True)
    shutil.copytree(args.challenger, root / "studies" / spec["archived_challenger_study_id"], symlinks=True)
    (root / "external").mkdir(mode=0o700)
    for name, path in (("toxic-chat_annotation_test.csv", args.toxic_test), ("xstest_prompts.csv", args.xstest)):
        shutil.copyfile(path, root / "external" / name)
    dump(root / "external/provenance.json", spec["external_regression_files"])
    (root / "wheel").mkdir(mode=0o700)
    shutil.copyfile(args.wheel, root / "wheel" / args.wheel.name)
    shutil.copyfile(args.source_archive, root / "release-source.tar")
    guide = (code / "README.md").read_text()
    guide += ("\n## Native reproduction\n\nUse the pinned s390x runtime image and the private bundle.\n"
              "Run from this extracted bundle directory in the qualified native environment:\n\n"
              "```bash\npython code/scripts/reproduce_selected.py --bundle . "
              "--output ../reproduction-01 --release-source-commit " + args.source_commit + "\n```\n")
    (root / "README.md").write_text(guide, encoding="utf-8")
    dump(root / "package.json", {"release_id": spec["release_id"], "source_commit": args.source_commit,
        "selected_study": "studies/" + spec["selected_study_id"],
        "archived_challenger": "studies/" + spec["archived_challenger_study_id"],
        "source_archive_sha256": sha(args.source_archive), "wheel_sha256": sha(args.wheel),
        "runtime_image_id": spec["runtime_image_id"], "private_package": True,
        "raw_data_and_model_weights_must_not_be_published": True})
    names = ["numpy", "scipy", "scikit-learn", "pandas", "joblib", "torch", "transformers", "pytest", "setuptools", "wheel"]
    dump(root / "runtime-environment.json", {"architecture": platform.machine(),
        "python": platform.python_version(), "system": platform.system(),
        "packages": {name: importlib.metadata.version(name) for name in names}})
    print(json.dumps({"status": "assembled_private_candidate", "release_id": spec["release_id"]}))


def seal(args):
    root = args.bundle.resolve()
    qualification = json.loads((args.qualification / "qualification-receipt.json").read_text())
    reproduction = json.loads((args.qualification / "reproduction/reproduction-receipt.json").read_text())
    package = json.loads((root / "package.json").read_text())
    if qualification["status"] != "passed" or reproduction["status"] != "passed":
        raise ValueError("release_qualification_required")
    if qualification["architecture"] != "s390x" or qualification["source_commit"] != package["source_commit"]:
        raise ValueError("release_qualification_binding_mismatch")
    shutil.copytree(args.qualification, root / "qualification")
    spec = json.loads((root / "code/src/release/selected.json").read_text())
    for field, expected in (("selected_study", spec["study_freeze_sha256"]),
                            ("archived_challenger", spec["archived_challenger_freeze_sha256"])):
        study = root / package[field]
        verify_tree(study, study / "freeze-manifest.json", expected)
    files, links = {}, {}
    for p in sorted(root.rglob("*")):
        name = str(p.relative_to(root))
        if p.is_symlink():
            if not p.resolve().is_relative_to(root):
                raise ValueError("bundle_link_escapes_root")
            links[name] = str(p.readlink())
        elif p.is_file():
            files[name] = sha(p)
    dump(root / "bundle-manifest.json", {"files": files, "links": links})
    archive = root.with_suffix(".tar.gz")
    if archive.exists():
        raise FileExistsError("release_archive_exists")
    with tarfile.open(archive, "w:gz", compresslevel=6, dereference=False) as output:
        output.add(root, arcname=root.name)
    archive.chmod(0o400)
    receipt = {"status": "sealed", "release_id": package["release_id"], "source_commit": package["source_commit"],
               "archive": archive.name, "bytes": archive.stat().st_size, "sha256": sha(archive),
               "manifest_files": len(files), "private": True, "native_qualification": qualification["status"]}
    dump(archive.parent / (root.name + "-receipt.json"), receipt)
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    assemble_parser = sub.add_parser("assemble")
    for name in ("selection", "study", "challenger", "source-archive", "wheel", "toxic-test", "xstest", "output"):
        assemble_parser.add_argument("--" + name, required=True, type=Path)
    assemble_parser.add_argument("--source-commit", required=True)
    seal_parser = sub.add_parser("seal")
    seal_parser.add_argument("--bundle", required=True, type=Path)
    seal_parser.add_argument("--qualification", required=True, type=Path)
    args = parser.parse_args()
    (assemble if args.command == "assemble" else seal)(args)


if __name__ == "__main__":
    main()
