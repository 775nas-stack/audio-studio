from pathlib import Path
import crepe, shutil

model_dir = Path(crepe.__file__).parent / "models"
target = Path("models/melody")
target.mkdir(parents=True, exist_ok=True)

for f in model_dir.glob("*"):
    shutil.copy(str(f), str(target))
    print("Copied:", f, "->", target)
