#!/bin/bash
# cleanup_results.sh - Manage and archive simulation results
#
# Usage: ./cleanup_results.sh [COMMAND]
#        COMMAND: archive, clean, summary (default: summary)

COMMAND="${1:-summary}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

case $COMMAND in
    summary)
        echo "=== Results Summary ==="
        echo ""
        
        # Count files
        CRIM_COUNT=$(ls -1 results/nav_results_*.mat 2>/dev/null | grep -v smoke | wc -l)
        SMOKE_COUNT=$(ls -1 results/smoke_nav_results_*.mat 2>/dev/null | wc -l)
        TOTAL_COUNT=$((CRIM_COUNT + SMOKE_COUNT))
        
        echo "Crimaldi results: $CRIM_COUNT files"
        echo "Smoke results: $SMOKE_COUNT files"
        echo "Total files: $TOTAL_COUNT"
        
        if [ $TOTAL_COUNT -gt 0 ]; then
            # Calculate size
            TOTAL_SIZE=$(du -sh results/*.mat 2>/dev/null | tail -1 | awk '{print $1}')
            echo "Total size: $TOTAL_SIZE"
            
            # Show date range
            OLDEST=$(ls -t results/*nav_results_*.mat 2>/dev/null | tail -1)
            NEWEST=$(ls -t results/*nav_results_*.mat 2>/dev/null | head -1)
            
            if [ -n "$OLDEST" ]; then
                echo ""
                echo "Date range:"
                echo "  Oldest: $(stat -c %y "$OLDEST" 2>/dev/null || stat -f "%Sm" "$OLDEST" 2>/dev/null | head -1)"
                echo "  Newest: $(stat -c %y "$NEWEST" 2>/dev/null || stat -f "%Sm" "$NEWEST" 2>/dev/null | head -1)"
            fi
        fi
        
        echo ""
        echo "Commands:"
        echo "  $0 archive   # Archive all results"
        echo "  $0 clean     # Remove all results (with confirmation)"
        ;;
        
    archive)
        if [ $(ls -1 results/*nav_results_*.mat 2>/dev/null | wc -l) -eq 0 ]; then
            echo "No results to archive"
            exit 0
        fi
        
        ARCHIVE_DIR="results/archive_${TIMESTAMP}"
        echo "=== Archiving Results ==="
        echo "Archive directory: $ARCHIVE_DIR"
        echo ""
        
        # Count files to archive
        CRIM_COUNT=$(ls -1 results/nav_results_*.mat 2>/dev/null | grep -v smoke | wc -l)
        SMOKE_COUNT=$(ls -1 results/smoke_nav_results_*.mat 2>/dev/null | wc -l)
        
        echo "Files to archive:"
        echo "  Crimaldi: $CRIM_COUNT"
        echo "  Smoke: $SMOKE_COUNT"
        echo ""
        
        read -p "Proceed with archive? (yes/no): " CONFIRM
        if [ "$CONFIRM" != "yes" ]; then
            echo "Cancelled"
            exit 0
        fi
        
        # Create archive directory
        mkdir -p "$ARCHIVE_DIR"
        
        # Move files
        echo "Moving files..."
        mv results/nav_results_*.mat "$ARCHIVE_DIR/" 2>/dev/null
        mv results/smoke_nav_results_*.mat "$ARCHIVE_DIR/" 2>/dev/null
        
        # Create summary
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
        if command -v matlab >/dev/null 2>&1; then
            echo "Generating summary statistics..."
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
        fi
        
        echo ""
        echo "✓ Archive complete: $ARCHIVE_DIR"
        echo "✓ Results directory is now empty"
        ;;
        
    clean)
        if [ $(ls -1 results/*nav_results_*.mat 2>/dev/null | wc -l) -eq 0 ]; then
            echo "No results to clean"
            exit 0
        fi
        
        echo "=== Clean Results ==="
        echo ""
        echo "WARNING: This will permanently delete all result files!"
        echo ""
        
        # Show what will be deleted
        CRIM_COUNT=$(ls -1 results/nav_results_*.mat 2>/dev/null | grep -v smoke | wc -l)
        SMOKE_COUNT=$(ls -1 results/smoke_nav_results_*.mat 2>/dev/null | wc -l)
        
        echo "Files to delete:"
        echo "  Crimaldi: $CRIM_COUNT"
        echo "  Smoke: $SMOKE_COUNT"
        echo ""
        
        read -p "Type 'DELETE' to confirm: " CONFIRM
        if [ "$CONFIRM" != "DELETE" ]; then
            echo "Cancelled"
            exit 0
        fi
        
        # Delete files
        echo "Deleting files..."
        rm -f results/nav_results_*.mat
        rm -f results/smoke_nav_results_*.mat
        
        echo "✓ Results cleaned"
        ;;
        
    *)
        echo "Usage: $0 {summary|archive|clean}"
        echo ""
        echo "  summary - Show current results summary"
        echo "  archive - Move results to timestamped archive"
        echo "  clean   - Delete all results (requires confirmation)"
        ;;
esac