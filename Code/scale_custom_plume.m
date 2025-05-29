function out_meta = scale_custom_plume(in_meta, out_video, out_meta)
%SCALE_CUSTOM_PLUME Rescale plume video and update metadata.
%  OUT_META = SCALE_CUSTOM_PLUME(IN_META, OUT_VIDEO, OUT_META) loads the
%  plume movie referenced by IN_META, rescales its intensity range to match
%  the CRIM dataset and saves it to OUT_VIDEO. A copy of the metadata is
%  written to OUT_META with the output directory and filename updated and
%  a flag 'scaled_to_crim' set to true. The path to OUT_META is returned.
%
%  Example:
%      scaled_meta = scale_custom_plume('meta.yaml', 'scaled.avi', ...
%                                       'scaled.yaml');
%      plume = load_custom_plume(scaled_meta);
%
%  See also: load_custom_plume, load_plume_video, plume_intensity_stats,
%  rescale_plume_range

arguments
    in_meta (1,:) char
    out_video (1,:) char
    out_meta (1,:) char
end

info = load_config(in_meta);
video_path = fullfile(info.output_directory, info.output_filename);
px_per_mm = 1 / info.vid_mm_per_px;
frame_rate = info.fps;

plume = load_plume_video(video_path, px_per_mm, frame_rate);
% Store the original intensity range before any rescaling
origMin = min(plume.data(:));
origMax = max(plume.data(:));

stats = plume_intensity_stats();
scaled = rescale_plume_range(plume.data, stats.CRIM.min, stats.CRIM.max);

% Register intensity range for the input video if not present
registry_path = fullfile('configs', 'plume_registry.yaml');
if exist(registry_path, 'file')
    registry = load_yaml(registry_path);
else
    registry = struct();
end
if ~isfield(registry, info.output_filename)
    update_plume_registry(info.output_filename, origMin, origMax, registry_path);
end

% store movie in 0..1 so load_custom_plume rescales correctly
scaled01 = rescale_plume_range(scaled, 0, 1);

vw = VideoWriter(out_video);
open(vw);
for k = 1:size(scaled01, 3)
    writeVideo(vw, scaled01(:,:,k));
end
close(vw);

newInfo = info;
[out_dir, out_name, out_ext] = fileparts(out_video);
if isempty(out_dir)
    out_dir = '.';
end
% Register the output video with CRIM range
update_plume_registry([out_name out_ext], stats.CRIM.min, stats.CRIM.max, registry_path);
newInfo.output_directory = out_dir;
newInfo.output_filename = [out_name out_ext];
newInfo.scaled_to_crim = true;

try
    if exist('yamlwrite', 'file') == 2
        yamlwrite(out_meta, newInfo);
    else
        fid = fopen(out_meta, 'w');
        fwrite(fid, jsonencode(newInfo));
        fclose(fid);
    end
catch ME
    warning('scale_custom_plume:WriteFailed', 'Failed to save YAML: %s', ...
            ME.message);
end

out_meta = out_meta;
end

