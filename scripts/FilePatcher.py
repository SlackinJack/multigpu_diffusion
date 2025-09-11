import copy
from pathlib import Path


def patch(patches):
    for entry in patches:
        file_name = entry.get("file_name")
        try:
            # read file
            with open(file_name, "r") as file:
                orig = file.read()
                out = copy.deepcopy(orig)

            # replace strings
            for replacement in entry.get("replace"):
                out = out.replace(replacement.get("from"), replacement.get("to"))

            # overwrite file
            with open(file_name, "w") as file:
                file.write(out)

            if out != orig:
                print(f"File patched: {file_name}")
        except Exception as e:
            print(f"Failed to patch file: {file_name}")
            print(str(e))


def patch_any(root, target, patches):
    result = list(Path(root).rglob(target))
    for path in result:
        file_name = path.as_posix()
        for entry in patches:
            patch([{"file_name": file_name, "replace": entry.get("replace")}])


def patch_flexible(patches):
    for entry in patches:
        result = input(f'{entry.get("question")}: ')
        if result.lower() in entry.get("proceed"):
            patch([{"file_name": entry.get("file_name"), "replace": entry.get("replace")}])
        else:
            print("Skipping this patch.")
