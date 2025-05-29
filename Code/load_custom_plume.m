function plume = load_custom_plume(metadata_path)
%LOAD_CUSTOM_PLUME Load plume video based on metadata YAML.
%   PLUME = LOAD_CUSTOM_PLUME(METADATA_PATH) reads the metadata YAML file
%   and loads the corresponding plume video using LOAD_PLUME_VIDEO.
%   The metadata file must contain the fields:
%       output_directory - directory containing the video file
%       output_filename  - name of the video file
%       vid_mm_per_px    - millimeters per pixel for the video
%       fps              - frame rate of the processed video
%
%   The returned structure is the same as produced by LOAD_PLUME_VIDEO. The
%   plume data is rescaled to the CRIM intensity range using
%   RESCALE_PLUME_RANGE.
%
%   Example:
%       plume = load_custom_plume('meta.yaml');

info = load_config(metadata_path);

px_per_mm = 1 / info.vid_mm_per_px;
frame_rate = info.fps;

use_h5 = false;
if isfield(info, 'output_h5')
    h5_path = fullfile(info.output_directory, info.output_h5);
    use_h5 = true;
elseif endsWith(info.output_filename, {'.h5', '.hdf5'}, 'IgnoreCase', true)
    h5_path = fullfile(info.output_directory, info.output_filename);
    use_h5 = true;
else
    video_path = fullfile(info.output_directory, info.output_filename);
end

if use_h5
    plume = load_plume_hdf5(h5_path, px_per_mm, frame_rate);
else
    plume = load_plume_video(video_path, px_per_mm, frame_rate);
end

stats = plume_intensity_stats();
plume.data = rescale_plume_range(plume.data, stats.CRIM.min, stats.CRIM.max);
end
