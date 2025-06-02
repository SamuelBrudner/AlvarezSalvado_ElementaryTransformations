#!/bin/bash
# Fix Elifenavmodel_bilateral.m to use h5read for video case

cp Code/Elifenavmodel_bilateral.m Code/Elifenavmodel_bilateral.m.bak

# Replace all plume.data references in video case
sed -i '/case.*video/,/^[[:space:]]*end[[:space:]]*$/{
    s/tind = mod(i-1, size(plume\.data,3)) + 1;/tind = mod(i-1, plume.dims(3)) + 1;/
    s/xind = round(10\*x(i,:)\*plume\.px_per_mm) + round(size(plume\.data,2)\/2);/xind = round(10*x(i,:)*plume.px_per_mm) + round(plume.dims(1)\/2);/
    s/xLind = round(10\*xL(i,:)\*plume\.px_per_mm) + round(size(plume\.data,2)\/2);/xLind = round(10*xL(i,:)*plume.px_per_mm) + round(plume.dims(1)\/2);/
    s/xRind = round(10\*xR(i,:)\*plume\.px_per_mm) + round(size(plume\.data,2)\/2);/xRind = round(10*xR(i,:)*plume.px_per_mm) + round(plume.dims(1)\/2);/
    s/out_of_plume = union(union(find(xind<1),find(xind>size(plume\.data,2))), \\/out_of_plume = union(union(find(xind<1),find(xind>plume.dims(1))), \\/
    s/union(find(yind<1),find(yind>size(plume\.data,1))));/union(find(yind<1),find(yind>plume.dims(2))));/
    s/out_of_plumeL = union(union(find(xLind<1),find(xLind>size(plume\.data,2))), \\/out_of_plumeL = union(union(find(xLind<1),find(xLind>plume.dims(1))), \\/
    s/union(find(yLind<1),find(yLind>size(plume\.data,1))));/union(find(yLind<1),find(yLind>plume.dims(2))));/
    s/out_of_plumeR = union(union(find(xRind<1),find(xRind>size(plume\.data,2))), \\/out_of_plumeR = union(union(find(xRind<1),find(xRind>plume.dims(1))), \\/
    s/union(find(yRind<1),find(yRind>size(plume\.data,1))));/union(find(yRind<1),find(yRind>plume.dims(2))));/
    s/odor(i,it) = plume\.data(yind(it), xind(it), tind);/odor(i,it) = max(0,h5read(plume.filename,plume.dataset,[xind(it) yind(it) tind],[1 1 1]));/
    s/odorL(i,it) = plume\.data(yLind(it), xLind(it), tind);/odorL(i,it) = max(0,h5read(plume.filename,plume.dataset,[xLind(it) yLind(it) tind],[1 1 1]));/
    s/odorR(i,it) = plume\.data(yRind(it), xRind(it), tind);/odorR(i,it) = max(0,h5read(plume.filename,plume.dataset,[xRind(it) yRind(it) tind],[1 1 1]));/
}' Code/Elifenavmodel_bilateral.m

echo "Fixed Elifenavmodel_bilateral.m"
