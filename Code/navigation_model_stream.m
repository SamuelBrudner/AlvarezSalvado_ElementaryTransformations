function out = navigation_model_vec_stream(triallength, environment, plotting, ...
                                           ntrials, vr, params)
% STREAMING variant of navigation_model_vec:
%  • 'vr' is a VideoReader handle
%  • every frame is read on-the-fly; nothing huge is pre-allocated

if nargin < 6,  params = struct(); end            % unchanged bits …  
if nargin < 4,  ntrials = 1;        end

% ─── all the parameter declarations stay the same ─── %
% (beta, tau_Aon, … kdown)

pxscale = 1/params.px_per_mm;        % keep the existing scale logic
tscale  = vr.FrameRate / 50;         % 15/50 folding handled later

% … the 15 Hz rescaling clause is unchanged …

% ─── allocate ONLY what we truly need in RAM ─── %
x      = nan(triallength,ntrials);         % positions etc.
y      = nan(triallength,ntrials);
heading= nan(triallength,ntrials);
odor   = zeros(1,ntrials,'single');        % *** just one row! ***
% same for ON/OFF; keep them 1×n and roll them each timestep
Aon = zeros(1,ntrials);  Aoff = zeros(1,ntrials);
ON  = zeros(1,ntrials);  R    = zeros(1,ntrials); Rh = zeros(1,ntrials);

% ---- initial conditions identical to original ---- %

% convenience: store basic frame geometry once
H = vr.Height;  W = vr.Width;

for i = 1:triallength
    % grab next video frame, loop if necessary
    if ~hasFrame(vr)
        if isfield(params,'loop') && params.loop
            vr.CurrentTime = 0;          % rewind
        else
            % leave odor = 0 for rest of trial
            odor(:) = 0;
            % continue through dynamics with zero stimulus
            % -> same effect as plume running out
        end
    end
    if hasFrame(vr)
        frame = rgb2gray(readFrame(vr));         % uint8
        frame = im2single(frame);                % 0-1, fast
    end

    % -------- sample odor for every fly --------------
    % (same indexing math as before, but using ‘frame’ instead of
    %  plume.data(:,:,tind))
    % NB: we only ever touch one pixel per fly per time step.
    for f = 1:ntrials
        xi = round(10*x(i,f)*params.px_per_mm)+1;
        yi = round(-10*y(i,f)*params.px_per_mm)+1;
        if xi<1 || xi>W || yi<1 || yi>H
            odor(f) = 0;
        else
            odor(f) = frame(yi,xi);
        end
    end

    % ------- identical model math from here down ------- %
    % *Use odor(f) instead of odor(i,f), because we don’t have a T×N array*

    % Adaptation, ON/OFF filters, turning, position update …
    % just replace every “odor(i, :)” with “odor(:)”.
    % (Same for ON/OFF if you keep them as 1×n vectors.)

end

% ---- pack ‘out’ the same way as before, but build odor as sparse ---- %
out.environment = 'video';
out.x      = x;        out.y   = y;
out.theta  = heading;
out.odor   = [];       % not saved to avoid ballooning file size
out.ON     = [];       % idem
out.OFF    = [];
out.turn   = [];       % you CAN still save turn if you like
out.params = params;
end