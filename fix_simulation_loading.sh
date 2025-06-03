#!/bin/bash
# fix_simulation_loading.sh - Add simulation config loading to get_plume_file.m

FILE="Code/get_plume_file.m"

echo "=== Fixing simulation config loading ==="
echo ""

# Backup
cp "$FILE" "${FILE}.backup_sim"
echo "✓ Backed up to ${FILE}.backup_sim"

# Find where to insert (after dataset_name block)
LINE_NUM=$(grep -n "plume_config.dataset_name = cfg.data_path.dataset_name" "$FILE" | cut -d: -f1)

if [ -z "$LINE_NUM" ]; then
    echo "✗ Could not find insertion point"
    exit 1
fi

# Insert point is 2 lines after (after the 'end' statement)
INSERT_LINE=$((LINE_NUM + 2))

# Create the code to insert
cat > /tmp/sim_code.txt << 'EOF'
    
    % Load simulation parameters
    if isfield(cfg, 'simulation')
        if isfield(cfg.simulation, 'duration_seconds')
            plume_config.simulation.duration_seconds = cfg.simulation.duration_seconds;
        end
        if isfield(cfg.simulation, 'comment')
            plume_config.simulation.comment = cfg.simulation.comment;
        end
    end
EOF

# Insert the code
head -n $INSERT_LINE "$FILE" > /tmp/new_file.m
cat /tmp/sim_code.txt >> /tmp/new_file.m
tail -n +$((INSERT_LINE + 1)) "$FILE" >> /tmp/new_file.m

# Replace the file
mv /tmp/new_file.m "$FILE"

echo "✓ Added simulation config loading"
echo ""
echo "Verify the change:"
echo "grep -A8 'Load simulation' $FILE"
echo ""
echo "Then test: ./run_test.sh"