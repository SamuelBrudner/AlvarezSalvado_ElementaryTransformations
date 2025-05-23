import json

__all__ = ["safe_load", "safe_dump"]

def safe_load(text):
    try:
        return json.loads(text)
    except Exception:
        data = {}
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip().strip('"\'')
            if value.lower() in ("true", "false"):
                data[key] = value.lower() == "true"
            else:
                try:
                    if '.' in value:
                        data[key] = float(value)
                    else:
                        data[key] = int(value)
                except ValueError:
                    data[key] = value
        return data

def safe_dump(data):
    return json.dumps(data)
