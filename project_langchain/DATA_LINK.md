# Data Directory Setup

The `data/` directory should point to `../project/data/` to reuse the processed course data.

## On Linux/Mac (Minerva)

Create a symlink:
```bash
cd project_langchain
ln -s ../project/data data
```

## On Windows

You can either:
1. Create a directory junction:
   ```powershell
   New-Item -ItemType Junction -Path "project_langchain\data" -Target "project\data"
   ```

2. Or copy the data directory (if you want independent copies):
   ```powershell
   Copy-Item -Recurse project\data project_langchain\data
   ```

The setup script expects the processed data at `data/processed/courses_clean.json`.

