function plume = load_custom_plume(metadata_path)
%LOAD_CUSTOM_PLUME Load plume video based on metadata YAML.
%   PLUME = LOAD_CUSTOM_PLUME(METADATA_PATH) reads the metadata YAML file
%   and loads the corresponding plume video using LOAD_PLUME_VIDEO.
%   The metadata file must contain the fields:
%       output_directory - directory containing the video file
%       output_filename  - name of the video file
%       vid_mm_per_px    - millimeters per pixel for the video
%       fps              - frame rate of the processed video
%       scaled_to_crim   - (optional) set true if video is already scaled
%
%   The returned structure is the same as produced by LOAD_PLUME_VIDEO. The
%   plume data is rescaled to the CRIM intensity range using RESCALE_PLUME_RANGE
%   unless the optional metadata field 'scaled_to_crim' is true.
%
%   Example:
%       plume = load_custom_plume('meta.yaml');

info = load_config(metadata_path);

video_path = fullfile(info.output_directory, info.output_filename);
px_per_mm = 1 / info.vid_mm_per_px;
frame_rate = info.fps;

plume = load_plume_video(video_path, px_per_mm, frame_rate);

if ~isfield(info, 'scaled_to_crim') || ~info.scaled_to_crim
    stats = plume_intensity_stats();
    plume.data = rescale_plume_range(plume.data, stats.CRIM.min, stats.CRIM.max);
end
end
