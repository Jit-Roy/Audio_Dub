import json
import hashlib
from pathlib import Path
from typing import Any, Optional

def get_file_hash(filepath: Path, chunk_size: int = 8192) -> str:
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        return ""

class CacheManager:
    """Manages reading and writing to disk cache with standard formatting and hash invalidation."""
    def __init__(self, cache_dir: Path, input_file: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.input_hash = get_file_hash(input_file)
        self._validate_global_cache()
        
    def _get_manifest_path(self) -> Path:
        return self.cache_dir / "cache_manifest.json"
        
    def _validate_global_cache(self):
        manifest_path = self._get_manifest_path()
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                if manifest.get("input_hash") != self.input_hash:
                    # Hash changed, invalidate all
                    print("[Cache] Input file modification detected. Invalidating stale cache.")
                    self.clear_all()
                    self._save_manifest()
            except json.JSONDecodeError:
                self.clear_all()
                self._save_manifest()
        else:
            self.clear_all()
            self._save_manifest()
            
    def _save_manifest(self):
        with open(self._get_manifest_path(), "w", encoding="utf-8") as f:
            json.dump({"input_hash": self.input_hash}, f)
            
    def clear_all(self):
        for item in self.cache_dir.iterdir():
            if item.is_file() and item.name != "cache_manifest.json":
                item.unlink()
            elif item.is_dir():
                import shutil
                shutil.rmtree(item)
    
    def get_path(self, key: str) -> Path:
        return self.cache_dir / key
    
    def exists(self, key: str) -> bool:
        return self.get_path(key).exists()
    
    def load_json(self, key: str, default: Any = None) -> Any:
        path = self.get_path(key)
        if not path.exists():
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Cache] Error loading {key}: {e}")
            return default
            
    def save_json(self, key: str, data: Any):
        path = self.get_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
    def invalidate(self, key: str):
        path = self.get_path(key)
        if path.exists():
            if path.is_file():
                path.unlink()
            else:
                import shutil
                shutil.rmtree(path)
