#!/bin/bash
# Complete test data analysis workflow

echo "=== COMPLETE TEST DATA ANALYSIS ==="
echo

# Check environment
if [ ! -d "./dev_env" ]; then
    echo "Error: dev_env not found. Run ./setup_env.sh --dev first"
    exit 1
fi

# Step 1: Export test data
echo "Step 1: Exporting test data..."
cat > export_test_data.py << 'EOF'
# (paste the python_export_test_data content here)
EOF

conda run --prefix ./dev_env python export_test_data.py
echo

# Step 2: Analyze exported data
echo "Step 2: Analyzing exported data..."
cat > analyze_test_data.py << 'EOF'
# (paste the analyze_exported_test_data content here)
EOF

conda run --prefix ./dev_env python analyze_test_data.py
echo

# Step 3: Monitor batch jobs (optional)
echo "Step 3: Batch job status..."
squeue -u $USER | head -10
echo

echo "=== ANALYSIS COMPLETE ==="
echo "Check these files for results:"
echo "  - test_trajectories_analysis.png"
echo "  - test_metrics_comparison.png"
echo "  - test_analysis_report.md"