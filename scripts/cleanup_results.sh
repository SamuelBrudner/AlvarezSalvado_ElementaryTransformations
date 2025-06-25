#!/bin/bash
# cleanup_results.sh - Manage and archive simulation results
#
# Usage: ./cleanup_results.sh [COMMAND] [OPTIONS]
#        COMMAND: archive, clean, summary (default: summary)
#        OPTIONS: -v, --verbose (enable detailed trace output)

# Parse verbose flag from arguments
VERBOSE=0
ORIGINAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Verbose logging enabled"
            shift
            ;;
        *)
            ORIGINAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore original arguments after verbose flag extraction
set -- "${ORIGINAL_ARGS[@]}"

COMMAND="${1:-summary}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

[[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Starting with command: $COMMAND"
[[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Timestamp: $TIMESTAMP"

case $COMMAND in
    summary)
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Executing summary command"
        echo "=== Results Summary ==="
        echo ""
        
        # Count files
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Counting Crimaldi result files"
        CRIM_COUNT=$(ls -1 results/nav_results_*.mat 2>/dev/null | grep -v smoke | wc -l)
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Found $CRIM_COUNT Crimaldi files"
        
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Counting Smoke result files"
        SMOKE_COUNT=$(ls -1 results/smoke_nav_results_*.mat 2>/dev/null | wc -l)
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Found $SMOKE_COUNT Smoke files"
        
        TOTAL_COUNT=$((CRIM_COUNT + SMOKE_COUNT))
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Total file count: $TOTAL_COUNT"
        
        echo "Crimaldi results: $CRIM_COUNT files"
        echo "Smoke results: $SMOKE_COUNT files"
        echo "Total files: $TOTAL_COUNT"
        
        if [ $TOTAL_COUNT -gt 0 ]; then
            # Calculate size
            [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Calculating total size of result files"
            TOTAL_SIZE=$(du -sh results/*.mat 2>/dev/null | tail -1 | awk '{print $1}')
            echo "Total size: $TOTAL_SIZE"
            [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Total size calculated: $TOTAL_SIZE"
            
            # Show date range
            [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Determining file date range"
            OLDEST=$(ls -t results/*nav_results_*.mat 2>/dev/null | tail -1)
            NEWEST=$(ls -t results/*nav_results_*.mat 2>/dev/null | head -1)
            
            if [ -n "$OLDEST" ]; then
                [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Oldest file: $OLDEST"
                [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Newest file: $NEWEST"
                echo ""
                echo "Date range:"
                echo "  Oldest: $(stat -c %y "$OLDEST" 2>/dev/null || stat -f "%Sm" "$OLDEST" 2>/dev/null | head -1)"
                echo "  Newest: $(stat -c %y "$NEWEST" 2>/dev/null || stat -f "%Sm" "$NEWEST" 2>/dev/null | head -1)"
            fi
        else
            [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: No result files found for analysis"
        fi
        
        echo ""
        echo "Commands:"
        echo "  $0 archive   # Archive all results"
        echo "  $0 clean     # Remove all results (with confirmation)"
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Summary command completed"
        ;;
        
    archive)
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Executing archive command"
        
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Checking for files to archive"
        if [ $(ls -1 results/*nav_results_*.mat 2>/dev/null | wc -l) -eq 0 ]; then
            echo "No results to archive"
            [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: No files found, exiting archive operation"
            exit 0
        fi
        
        ARCHIVE_DIR="results/archive_${TIMESTAMP}"
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Archive directory set to: $ARCHIVE_DIR"
        
        echo "=== Archiving Results ==="
        echo "Archive directory: $ARCHIVE_DIR"
        echo ""
        
        # Count files to archive
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Counting files for archive operation"
        CRIM_COUNT=$(ls -1 results/nav_results_*.mat 2>/dev/null | grep -v smoke | wc -l)
        SMOKE_COUNT=$(ls -1 results/smoke_nav_results_*.mat 2>/dev/null | wc -l)
        
        echo "Files to archive:"
        echo "  Crimaldi: $CRIM_COUNT"
        echo "  Smoke: $SMOKE_COUNT"
        echo ""
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Archive counts - Crimaldi: $CRIM_COUNT, Smoke: $SMOKE_COUNT"
        
        read -p "Proceed with archive? (yes/no): " CONFIRM
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: User confirmation: $CONFIRM"
        
        if [ "$CONFIRM" != "yes" ]; then
            echo "Cancelled"
            [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Archive operation cancelled by user"
            exit 0
        fi
        
        # Create archive directory
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Creating archive directory: $ARCHIVE_DIR"
        mkdir -p "$ARCHIVE_DIR"
        
        # Move files
        echo "Moving files..."
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Moving Crimaldi result files to archive"
        mv results/nav_results_*.mat "$ARCHIVE_DIR/" 2>/dev/null
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Moving Smoke result files to archive"
        mv results/smoke_nav_results_*.mat "$ARCHIVE_DIR/" 2>/dev/null
        
        # Create summary
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Creating archive information file"
        cat > "$ARCHIVE_DIR/archive_info.txt" << EOF
Archive Information
==================
Date: $(date)
User: $USER@$(hostname)

Contents:
  Crimaldi results: $CRIM_COUNT files
  Smoke results: $SMOKE_COUNT files
  Total: $((CRIM_COUNT + SMOKE_COUNT)) files

Original location: $(pwd)/results/
EOF
        
        # Quick analysis if MATLAB available
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Checking for MATLAB availability"
        if command -v matlab >/dev/null 2>&1; then
            echo "Generating summary statistics..."
            [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: MATLAB found, generating summary statistics"
            matlab -batch "
                cd('$(pwd)');
                archive_dir = '$ARCHIVE_DIR';
                
                % Analyze Crimaldi
                crim_files = dir(fullfile(archive_dir, 'nav_results_*.mat'));
                crim_files = crim_files(~contains({crim_files.name}, 'smoke'));
                crim_success = [];
                
                for i = 1:min(length(crim_files), 20)
                    try
                        d = load(fullfile(archive_dir, crim_files(i).name));
                        if isfield(d.out, 'successrate')
                            crim_success(end+1) = d.out.successrate * 100;
                        end
                    catch
                    end
                end
                
                % Analyze Smoke
                smoke_files = dir(fullfile(archive_dir, 'smoke_nav_results_*.mat'));
                smoke_success = [];
                
                for i = 1:min(length(smoke_files), 20)
                    try
                        d = load(fullfile(archive_dir, smoke_files(i).name));
                        if isfield(d.out, 'successrate')
                            smoke_success(end+1) = d.out.successrate * 100;
                        end
                    catch
                    end
                end
                
                % Write summary
                fid = fopen(fullfile(archive_dir, 'summary_stats.txt'), 'w');
                fprintf(fid, 'Summary Statistics\n');
                fprintf(fid, '==================\n\n');
                
                if ~isempty(crim_success)
                    fprintf(fid, 'Crimaldi (n=%d):\n', length(crim_success));
                    fprintf(fid, '  Mean success: %.1f%%\n', mean(crim_success));
                    fprintf(fid, '  Std dev: %.1f%%\n', std(crim_success));
                    fprintf(fid, '  Range: %.1f%% - %.1f%%\n\n', min(crim_success), max(crim_success));
                end
                
                if ~isempty(smoke_success)
                    fprintf(fid, 'Smoke (n=%d):\n', length(smoke_success));
                    fprintf(fid, '  Mean success: %.1f%%\n', mean(smoke_success));
                    fprintf(fid, '  Std dev: %.1f%%\n', std(smoke_success));
                    fprintf(fid, '  Range: %.1f%% - %.1f%%\n', min(smoke_success), max(smoke_success));
                end
                
                fclose(fid);
            " 2>/dev/null
            [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: MATLAB summary statistics generation completed"
        else
            [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: MATLAB not available, skipping summary statistics"
        fi
        
        echo ""
        echo "✓ Archive complete: $ARCHIVE_DIR"
        echo "✓ Results directory is now empty"
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Archive command completed successfully"
        ;;
        
    clean)
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Executing clean command"
        
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Checking for files to clean"
        if [ $(ls -1 results/*nav_results_*.mat 2>/dev/null | wc -l) -eq 0 ]; then
            echo "No results to clean"
            [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: No files found, exiting clean operation"
            exit 0
        fi
        
        echo "=== Clean Results ==="
        echo ""
        echo "WARNING: This will permanently delete all result files!"
        echo ""
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Warning displayed to user about permanent deletion"
        
        # Show what will be deleted
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Counting files for deletion confirmation"
        CRIM_COUNT=$(ls -1 results/nav_results_*.mat 2>/dev/null | grep -v smoke | wc -l)
        SMOKE_COUNT=$(ls -1 results/smoke_nav_results_*.mat 2>/dev/null | wc -l)
        
        echo "Files to delete:"
        echo "  Crimaldi: $CRIM_COUNT"
        echo "  Smoke: $SMOKE_COUNT"
        echo ""
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Deletion counts - Crimaldi: $CRIM_COUNT, Smoke: $SMOKE_COUNT"
        
        read -p "Type 'DELETE' to confirm: " CONFIRM
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: User confirmation for deletion: $CONFIRM"
        
        if [ "$CONFIRM" != "DELETE" ]; then
            echo "Cancelled"
            [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Clean operation cancelled by user"
            exit 0
        fi
        
        # Delete files
        echo "Deleting files..."
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Deleting Crimaldi result files"
        rm -f results/nav_results_*.mat
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Deleting Smoke result files"
        rm -f results/smoke_nav_results_*.mat
        
        echo "✓ Results cleaned"
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Clean command completed successfully"
        ;;
        
    *)
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Invalid command provided: $COMMAND"
        echo "Usage: $0 {summary|archive|clean} [OPTIONS]"
        echo ""
        echo "  summary - Show current results summary"
        echo "  archive - Move results to timestamped archive"
        echo "  clean   - Delete all results (requires confirmation)"
        echo ""
        echo "Options:"
        echo "  -v, --verbose  Enable detailed trace output"
        [[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Usage information displayed"
        ;;
esac

[[ $VERBOSE -eq 1 ]] && echo "[$(date)] cleanup_results.sh: Script execution completed"