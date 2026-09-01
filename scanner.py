import os, glob

def check_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    print(f"\n--- Checking {path} ---")
    for i, line in enumerate(lines):
        line_num = i + 1
        # BUGS
        if "gguf_init_from_file" in line or "fopen" in line or "open(" in line:
            print(f"BUGS_FILE {path}:{line_num} | {line.strip()}")
        if "malloc" in line or "new " in line:
            if not "delete" in "".join(lines) and not "free" in "".join(lines):
                print(f"BUGS_MEM {path}:{line_num} | {line.strip()}")
        
        # PERFORMANCE
        if "for (" in line or "while (" in line:
            if "get_f32" in line or "set_f32" in line or "memcpy" in line:
                print(f"PERF_LOOP {path}:{line_num} | {line.strip()}")
        
        # API/BUILD
        if "pragma once" not in "".join(lines) and path.endswith(".h"):
            print(f"API_HEADER {path}:{line_num} | Missing pragma once")
            break

for f in glob.glob("src/**/*.cpp", recursive=True) + glob.glob("src/**/*.h", recursive=True):
    check_file(f)
