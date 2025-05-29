%% quick_stream_test.m  ── run this from the repo root
% 1. Point a VideoReader at the plume movie
vr = VideoReader(fullfile('data','smoke_1a_orig_backgroundsubtracted.avi'));

% 2. Tell the model how to interpret that video
params = struct( ...
    'px_per_mm',  6.536, ...           % spatial scale
    'frame_rate', vr.FrameRate );      % should be 60 for this clip

% 3. Choose a triallength shorter than, equal to, or longer than the movie.
%    (If longer, the code will loop the plume automatically.)
triallength = 3 * vr.FrameRate;        % 3-s trial   ⇒   180 time-steps

% 4. Call the streaming wrapper.
%    environment MUST be 'video'; plotting = 0 is fastest.
out = navigation_model_vec_stream( ...
          triallength, ...             % # of steps you want
          'video', ...                 % environment
          0, ...                       % plotting flag
          1, ...                       % ntrials (agents)
          vr, ...                      % VideoReader handle
          params );                    % extra parameters

%% 5. Sanity-check the output
fprintf('Frames seen:         %d\n', size(out.odor,1));
fprintf('Triallength argued:   %d\n', triallength);
fprintf('x range: %g … %g cm\n',  min(out.x(:)), max(out.x(:)));
fprintf('Odor range: %g … %g\n', min(out.odor(:)), max(out.odor(:)));