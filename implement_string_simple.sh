#!/bin/bash
# implement_string_simple.sh - Simple one-step implementation

echo "=== Implementing String-Based Duration ('config') ==="
echo ""

# Run the Python implementation
if [ -f "implement_string_duration.py" ]; then
    python3 implement_string_duration.py
else
    # Create it inline
    python3 << 'EOF'
import re, shutil, json
from pathlib import Path

# Update config
config_path = Path('configs/plumes/crimaldi_10cms_bounded.json')
with open(config_path, 'r') as f:
    config = json.load(f)
if 'simulation' not in config:
    config['simulation'] = {'duration_seconds': 240.0, 'comment': '4 minutes'}
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print("✓ Added duration to config")

# Modify navigation_model_vec.m
file_path = Path('Code/navigation_model_vec.m')
if not Path('Code/navigation_model_vec.m.backup').exists():
    shutil.copy(file_path, 'Code/navigation_model_vec.m.backup')

with open(file_path, 'r') as f:
    content = f.read()

if 'ischar(triallength)' not in content:
    insertion = """
% Handle string input for triallength
if ischar(triallength) || isstring(triallength)
    if strcmpi(triallength, 'config') || strcmpi(triallength, 'auto')
        fprintf('Reading duration from config file...\\\\n');
        [~, plume_config] = get_plume_file();
        
        % Get duration in seconds from config
        if isfield(plume_config, 'simulation') && isfield(plume_config.simulation, 'duration_seconds')
            duration_seconds = plume_config.simulation.duration_seconds;
        else
            duration_seconds = 240.0;  % Default 4 minutes
            fprintf('No simulation.duration_seconds in config, using default: %.1f seconds\\\\n', duration_seconds);
        end
        
        % Convert to samples based on environment
        switch lower(environment)
            case {'crimaldi', 'openlooppulse15', 'openlooppulsewb15'}
                frame_rate = 15;
            case {'gaussian', 'openlooppulse', 'openlooppulsewb'}
                frame_rate = 50;
            otherwise
                frame_rate = 15;
        end
        
        triallength = round(duration_seconds * frame_rate);
        fprintf('Using config duration: %.1f seconds = %d samples at %d Hz\\\\n', ...
                duration_seconds, triallength, frame_rate);
    else
        error('Invalid triallength string. Use ''config'' or a number.');
    end
else
    % Ensure triallength is numeric if not a string
    triallength = double(triallength);
end

"""
    pattern = r'(tic\s*\n)(% Scaling factors)'
    new_content = re.sub(pattern, r'\1' + insertion + r'\2', content)
    with open(file_path, 'w') as f:
        f.write(new_content)
    print("✓ Modified navigation_model_vec.m")

print("\n✓ Implementation complete!")
EOF
fi

# Quick test using temporary MATLAB file
echo ""
echo "Testing..."

TEMP_SCRIPT=$(mktemp /tmp/test_string_impl_XXXXXX.m)
cat > "$TEMP_SCRIPT" << 'EOF'
addpath(genpath('Code'));
try
    out = navigation_model_vec('config', 'gaussian', 0, 1);
    fprintf('✓ Success! String parameter works\n');
    fprintf('  Got %d samples from config\n', size(out.x,1));
catch ME
    fprintf('✗ Error: %s\n', ME.message);
end
exit
EOF

matlab -nodisplay -nosplash -nojvm -r "run('$TEMP_SCRIPT')" 2>&1 | grep -E "(✓|✗|Reading)" | head -10

rm -f "$TEMP_SCRIPT"

echo ""
echo "=== Done! ==="
echo ""
echo "Usage examples:"
echo '  navigation_model_vec(3600, "Crimaldi", 0, 10)      % Old way'
echo '  navigation_model_vec("config", "Crimaldi", 0, 10)  % New way!'
echo ""
echo "To change duration: edit configs/plumes/crimaldi_10cms_bounded.json"
echo '  "duration_seconds": 300.0  (for 5 minutes)'