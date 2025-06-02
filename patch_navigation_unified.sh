#!/bin/bash
# Patch navigation_model_vec_unified.m to use h5read for video case

# Replace the video case odor reading
sed -i '/case.*video.*$/,/case\|^end$/{
    s/tind = mod(i-1, size(plume\.data,3)) + 1;/tind = mod(i-1, plume.dims(3)) + 1;/
    s/xind = round(10\*x(i,:)\*plume\.px_per_mm) + round(size(plume\.data,2)\/2);/xind = round(10*x(i,:)*plume.px_per_mm) + round(plume.dims(1)\/2);/
    s/out_of_plume = union(union(find(xind<1),find(xind>size(plume\.data,2))),union(find(yind<1),find(yind>size(plume\.data,1))));/out_of_plume = union(union(find(xind<1),find(xind>plume.dims(1))),union(find(yind<1),find(yind>plume.dims(2))));/
    s/odor(i,it) = plume\.data(yind(it), xind(it), tind);/odor(i,it) = max(0,h5read(plume.filename,plume.dataset,[xind(it) yind(it) tind],[1 1 1]));/
}' Code/navigation_model_vec_unified.m

echo "Patched navigation_model_vec_unified.m"
