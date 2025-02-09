import sys
import json

data = json.load(sys.stdin)
for key, value in data.items():
    print(f"{key}:")
    print(value)
    print()
