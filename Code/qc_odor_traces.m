function qc_odor_traces(n_examples, n_bins)
%QC_ODOR_TRACES  Quality-control diagnostic for odor sensing.
%   QC_ODOR_TRACES() loads all *_nav_results_*.mat files in the project
%   `results/` directory, segregates them by plume environment (e.g.
%   Crimaldi, Smoke) and generates a figure per environment containing:
%
%     1) Example odor-intensity traces for a handful of agents.
%     2) Mean ± SEM odor intensity as a function of time.
%     3) Mean odor intensity as a function of distance from the source.
%
%   The figures are saved to:
%       results/<env>_odor_qc.png   (PNG, suitable for quick viewing)
%       results/<env>_odor_qc.pdf   (vector PDF for publication)
%
%   Optional arguments
%   ------------------
%   N_EXAMPLES (default = 5)   → number of example agents to plot.
%   N_BINS     (default = 30)  → number of distance bins.
%
%   Author: Cascade-generated, July 2025

arguments
    n_examples (1,1) double {mustBePositive, mustBeInteger} = 5
    n_bins     (1,1) double {mustBePositive, mustBeInteger} = 30
end

project_dir = fileparts(fileparts(mfilename('fullpath')));   % repo root
results_dir = fullfile(project_dir, "results");

if ~isfolder(results_dir)
    warning("qc_odor_traces: results/ directory does not exist – nothing to do.");
    return
end

files = dir(fullfile(results_dir, "*_nav_results_*.mat"));
if isempty(files)
    fprintf("qc_odor_traces: no result files found in %s\n", results_dir);
    return
end

% -------------------------------------------------------------------------
% Group result files by environment key (prefix before _nav_results_)
% -------------------------------------------------------------------------
by_env = containers.Map('KeyType', 'char', 'ValueType', 'any');
for f = files'
    m = regexp(f.name, "^(?<env>[A-Za-z]+)_nav_results_", 'names');
    if isempty(m)
        continue
    end
    key = lower(m.env);
    if ~isKey(by_env, key)
        by_env(key) = {fullfile(f.folder, f.name)};
    else
        by_env(key) = [by_env(key), {fullfile(f.folder, f.name)}]; %#ok<AGROW>
    end
end

if isempty(keys(by_env))
    warning("qc_odor_traces: could not parse environment keys from filenames.");
    return
end

% -------------------------------------------------------------------------
% Locate plume configs – used to obtain source position
% -------------------------------------------------------------------------
plume_cfg_dir = fullfile(project_dir, "configs", "plumes");

for env_key = keys(by_env)
    env = env_key{1};               % char scalar (lowercase)
    result_paths = by_env(env);

    %% --- Aggregate data across files ------------------------------------
    odor_all = [];
    x_all    = [];
    y_all    = [];
    frame_rate = NaN;

    for p = 1:numel(result_paths)
        S = load(result_paths{p}, 'out');
        if ~isfield(S, 'out')
            warning('File %s missing variable "out" – skipping', result_paths{p});
            continue
        end
        out = S.out;
        if ~isfield(out, 'odor') || isempty(out.odor)
            warning('File %s lacks out.odor – skipping', result_paths{p});
            continue
        end
        odor_all = [odor_all, double(out.odor)]; %#ok<*AGROW>
        x_all    = [x_all, double(out.x)];
        y_all    = [y_all, double(out.y)];

        if isnan(frame_rate)
            if isfield(out, 'frame_rate') && ~isempty(out.frame_rate)
                frame_rate = double(out.frame_rate);
            elseif isfield(out, 'frameRate')
                frame_rate = double(out.frameRate);
            else
                frame_rate = NaN;
            end
        end
    end

    if isempty(odor_all)
        warning('No odor data aggregated for environment %s', env);
        continue
    end

    n_frames = size(odor_all, 1);
    n_agents = size(odor_all, 2);

    if isnan(frame_rate) || frame_rate <= 0
        frame_rate = 1;  % fallback – units become frames
        time_vec = 0:n_frames-1;
        time_label = "Frame #";
    else
        time_vec = (0:n_frames-1) ./ frame_rate;
        time_label = "Time (s)";
    end

    %% --- Determine source position -------------------------------------
    src_x = 0.0; src_y = 0.0;  % default
    cfg_matches = dir(fullfile(plume_cfg_dir, sprintf('%s*.json', env)));
    if ~isempty(cfg_matches)
        try
            cfg = jsondecode(fileread(fullfile(cfg_matches(1).folder, cfg_matches(1).name)));
            if isfield(cfg, 'simulation') && isfield(cfg.simulation, 'source_position')
                sp = cfg.simulation.source_position;
                if all(isfield(sp, {"x_cm", "y_cm"}))
                    src_x = double(sp.x_cm);
                    src_y = double(sp.y_cm);
                end
            end
        catch ME
            warning('Could not parse config %s: %s', cfg_matches(1).name, ME.message);
        end
    end

    %% --- Compute distances ---------------------------------------------
    dist_all = sqrt((x_all - src_x).^2 + (y_all - src_y).^2);

    %% --- Statistics -----------------------------------------------------
    mean_odor_time = mean(odor_all, 2, 'omitnan');
    sem_odor_time  = std(odor_all, 0, 2, 'omitnan') ./ sqrt(n_agents);

    % Odor vs distance (bin)
    dist_vec  = dist_all(:);
    odor_vec  = odor_all(:);

    edges = linspace(min(dist_vec), max(dist_vec), n_bins+1);
    [~, bin_idx] = histc(dist_vec, edges);
    odor_dist_mean = accumarray(bin_idx(bin_idx>0), odor_vec(bin_idx>0), [n_bins 1], @mean, NaN);
    dist_centers  = (edges(1:end-1) + edges(2:end)) / 2;

    %% --- Plot -----------------------------------------------------------
    fig = figure('Visible', 'off', 'Color', 'w', 'Name', sprintf('Odor QC – %s', env));
    tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    % (1) Example traces --------------------------------------------------
    nexttile;
    ex_idx = 1:min(n_examples, n_agents);
    plot(time_vec, odor_all(:, ex_idx), 'LineWidth', 1.0);
    xlabel(time_label); ylabel('Odor intensity');
    title(sprintf('%s – Example odor traces (n=%d)', capitalize(env), numel(ex_idx)));

    % (2) Mean ± SEM odor vs time ---------------------------------------
    nexttile;
    fill([time_vec fliplr(time_vec)], [mean_odor_time.' - sem_odor_time.' fliplr(mean_odor_time.' + sem_odor_time.')], ...
         [0.2 0.5 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    hold on;
    plot(time_vec, mean_odor_time, 'b-', 'LineWidth', 1.5);
    xlabel(time_label); ylabel('Mean odor ± SEM');
    title('Average odor over time');

    % (3) Odor vs distance ----------------------------------------------
    nexttile([1 2]);
    plot(dist_centers, odor_dist_mean, 'k-', 'LineWidth', 1.5);
    xlabel('Distance from source (cm)'); ylabel('Mean odor');
    title('Odor vs distance'); grid on;

    %% --- Save -----------------------------------------------------------
    out_png = fullfile(results_dir, sprintf('%s_odor_qc.png', env));
    out_pdf = fullfile(results_dir, sprintf('%s_odor_qc.pdf', env));
    try
        saveas(fig, out_png);
        saveas(fig, out_pdf);
        fprintf('[QC] Saved odor QC figure → %s\n', out_png);
    catch ME
        warning('qc_odor_traces: could not save figure: %s', ME.message);
    end
    close(fig);
end
end

function s = capitalize(str)
%CAPITALIZE  Uppercase first letter, lowercase the rest (simple utility)
if isempty(str)
    s = str;
else
    s = [upper(str(1)) lower(str(2:end))];
end
end
