#!/bin/bash
# Add HPC fix directly to navigation models

for model in "Elifenavmodel_bilateral.m" "navigation_model_vec.m"; do
    if [ -f "Code/$model" ]; then
        # Check if already patched
        if grep -q "HPC path fix" "Code/$model"; then
            echo "✓ $model already patched"
        else
            # Find line after get_plume_file call
            LINE=$(grep -n "plume_filename = get_plume_file" "Code/$model" | cut -d: -f1 | head -1)
            if [ ! -z "$LINE" ]; then
                # Insert fix after that line
                NEXT_LINE=$((LINE + 1))
                sed -i "${NEXT_LINE}i\\            % HPC path fix\\
            if contains(plume_filename, '/vast/palmer/home.grace/snb6/')\\
                plume_filename = strrep(plume_filename, '/vast/palmer/home.grace/snb6/', '/home/snb6/');\\
            end\\
            % Also check environment override\\
            env_override = getenv('MATLAB_PLUME_FILE');\\
            if ~isempty(env_override) && exist(env_override, 'file')\\
                plume_filename = env_override;\\
            end" "Code/$model"
                echo "✓ Patched $model"
            fi
        fi
    fi
done
