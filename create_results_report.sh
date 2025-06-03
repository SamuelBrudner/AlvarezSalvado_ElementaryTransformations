#!/bin/bash
# create_matlab_report.sh - Generate results summary using MATLAB
#
# Usage: ./create_matlab_report.sh
#
# Analyzes all nav_results_*.mat files in results/ directory
# Creates a summary report with statistics

RESULTS_DIR="results"
REPORT_FILE="results_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "=== Navigation Model Results Summary ===" > $REPORT_FILE
echo "Generated: $(date)" >> $REPORT_FILE
echo "" >> $REPORT_FILE

# Count result files
N_FILES=$(ls -1 $RESULTS_DIR/nav_results_*.mat 2>/dev/null | wc -l)
echo "Total result files: $N_FILES" >> $REPORT_FILE
echo "" >> $REPORT_FILE

if [ $N_FILES -eq 0 ]; then
    echo "No result files found in $RESULTS_DIR/" >> $REPORT_FILE
    cat $REPORT_FILE
    exit 0
fi

# Create MATLAB analysis script
cat > temp_analyze_all.m << 'EOF'
% Temporary script to analyze all results
results_dir = 'results';
files = dir(fullfile(results_dir, 'nav_results_*.mat'));

fprintf('\nIndividual Results:\n');
fprintf('==================\n');

all_success_rates = [];
all_latencies = [];
total_agents = 0;

for i = 1:length(files)
    filepath = fullfile(files(i).folder, files(i).name);
    fprintf('\nFile: %s\n', files(i).name);
    
    try
        data = load(filepath);
        out = data.out;
        
        [n_samples, n_agents] = size(out.x);
        total_agents = total_agents + n_agents;
        
        fprintf('  Agents: %d\n', n_agents);
        
        if isfield(out, 'successrate')
            fprintf('  Success rate: %.1f%%\n', out.successrate * 100);
            all_success_rates(end+1) = out.successrate;
        end
        
        if isfield(out, 'latency')
            successful = ~isnan(out.latency);
            n_success = sum(successful);
            if n_success > 0
                mean_lat = mean(out.latency(successful));
                fprintf('  Mean latency: %.1f s (n=%d)\n', mean_lat, n_success);
                all_latencies = [all_latencies, out.latency(successful)];
            end
        end
    catch ME
        fprintf('  Error: %s\n', ME.message);
    end
end

fprintf('\n\nOverall Statistics:\n');
fprintf('==================\n');
fprintf('Total agents simulated: %d\n', total_agents);

if ~isempty(all_success_rates)
    fprintf('Mean success rate: %.1f%%\n', mean(all_success_rates) * 100);
    fprintf('Success rate range: %.1f%% - %.1f%%\n', ...
            min(all_success_rates)*100, max(all_success_rates)*100);
end

if ~isempty(all_latencies)
    fprintf('Overall mean latency: %.1f seconds\n', mean(all_latencies));
    fprintf('Fastest: %.1f s, Slowest: %.1f s\n', ...
            min(all_latencies), max(all_latencies));
    fprintf('Total successful navigations: %d\n', length(all_latencies));
end

exit;
EOF

# Run MATLAB analysis
echo "Analyzing results with MATLAB..." >&2
matlab -nodisplay -nosplash < temp_analyze_all.m 2>/dev/null | grep -v "^>>" | tail -n +11 >> $REPORT_FILE

# Clean up
rm -f temp_analyze_all.m

# Display report
echo "Report saved to: $REPORT_FILE"
echo ""
cat $REPORT_FILE