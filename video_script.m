plume = load_plume_video('data/smoke_1a_bgsub_raw.avi',6.536,60);
all_intensities = plume.data(:);
save('temp_intensities.mat','all_intensities')
fprintf('TEMP_MAT_FILE_SUCCESS:%s\n', which('temp_intensities.mat'));