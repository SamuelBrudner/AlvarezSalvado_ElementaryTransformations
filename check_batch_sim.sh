# Check how many simulations completed
echo "=== FULL BATCH RESULTS ==="
find data/raw -name "result.mat" 2>/dev/null | wc -l

# Count by condition
echo -e "\n=== Results by condition ==="
find data/raw -type d -name "*_*" -maxdepth 2 | while read dir; do
    count=$(find "$dir" -name "result.mat" 2>/dev/null | wc -l)
    condition=$(basename "$dir")
    echo "$condition: $count agents"
done

# Check total disk usage
echo -e "\n=== Disk usage ==="
du -sh data/raw

# Sample a few results to verify they're complete
echo -e "\n=== Sample results (first 5) ==="
find data/raw -name "result.mat" | head -5 | while read f; do
    size=$(ls -lh "$f" | awk '{print $5}')
    echo "$f ($size)"
done