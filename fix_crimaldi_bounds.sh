#!/bin/bash

# Fix navigation_model_vec.m
sed -i '/xind = round(10\*x(i,:)\/pxscale)+108;/a\            xind = max(1, min(216, xind)); % Clamp to valid range' Code/navigation_model_vec.m
sed -i '/yind = -round(10\*y(i,:)\/pxscale)+1;/a\            yind = max(1, min(406, yind)); % Clamp to valid range' Code/navigation_model_vec.m

# Fix Elifenavmodel_bilateral.m  
sed -i '/xind = round(10\*x(i,:)\/pxscale)+108;/a\            xind = max(1, min(216, xind)); % Clamp to valid range' Code/Elifenavmodel_bilateral.m
sed -i '/yind = -round(10\*y(i,:)\/pxscale)+1;/a\            yind = max(1, min(406, yind)); % Clamp to valid range' Code/Elifenavmodel_bilateral.m

# Also fix the bilateral model's right antenna indices
sed -i '/xRind = round((1\/L)\*rx)+xind;/a\                xRind = max(1, min(216, xRind)); % Clamp to valid range' Code/Elifenavmodel_bilateral.m
sed -i '/yRind = -round((1\/L)\*ry)+yind;/a\                yRind = max(1, min(406, yRind)); % Clamp to valid range' Code/Elifenavmodel_bilateral.m
sed -i '/xRind = round(rx\/(pxscale\/10))+xind;/a\                xRind = max(1, min(216, xRind)); % Clamp to valid range' Code/Elifenavmodel_bilateral.m
sed -i '/yRind = -round(ry\/(pxscale\/10))+yind;/a\                yRind = max(1, min(406, yRind)); % Clamp to valid range' Code/Elifenavmodel_bilateral.m

echo "Bounds checking added to navigation models"
