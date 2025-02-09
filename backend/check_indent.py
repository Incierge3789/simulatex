with open('data_manager.py', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines, 1):
        if '_cleanup' in line:
            print(f"Line {i}: {repr(line)}")
            # 前後の行も表示
            print(f"Previous: {repr(lines[i-2])}")
            print(f"Current:  {repr(line)}")
            print(f"Next:     {repr(lines[i])}")
