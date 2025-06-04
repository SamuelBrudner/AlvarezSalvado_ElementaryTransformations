function sim_params = load_simulation_params()
%LOAD_SIMULATION_PARAMS Load simulation parameters from config
%   Returns a structure with all simulation parameters

% Set defaults
sim_params = struct();
sim_params.success_radius = 2.0;  % cm
sim_params.init_x_range = [-8, 8];  % cm
sim_params.init_y_range = [-30, -25];  % cm
sim_params.n_agents_per_job = 10;
sim_params.source_x = 0;  % cm
sim_params.source_y = 0;  % cm

% Try to load from config
try
    paths = load_paths();
    config_path = paths.plume_config;
    cfg = jsondecode(fileread(config_path));
    
    if isfield(cfg, 'simulation')
        sim = cfg.simulation;
        
        % Success radius
        if isfield(sim, 'success_radius_cm')
            sim_params.success_radius = sim.success_radius_cm;
        end
        
        % Agent initialization
        if isfield(sim, 'agent_initialization')
            agent_init = sim.agent_initialization;
            if isfield(agent_init, 'x_range_cm')
                sim_params.init_x_range = agent_init.x_range_cm;
            end
            if isfield(agent_init, 'y_range_cm')
                sim_params.init_y_range = agent_init.y_range_cm;
            end
            if isfield(agent_init, 'n_agents_per_job')
                sim_params.n_agents_per_job = agent_init.n_agents_per_job;
            end
        end
        
        % Source position
        if isfield(sim, 'source_position')
            if isfield(sim.source_position, 'x_cm')
                sim_params.source_x = sim.source_position.x_cm;
            end
            if isfield(sim.source_position, 'y_cm')
                sim_params.source_y = sim.source_position.y_cm;
            end
        end
    end
    
    fprintf('Loaded simulation parameters from config\n');
catch ME
    warning('Could not load simulation parameters from config: %s', ME.message);
    fprintf('Using default simulation parameters\n');
end

% Display loaded parameters
fprintf('  Success radius: %.1f cm\n', sim_params.success_radius);
fprintf('  Init X range: [%.1f, %.1f] cm\n', sim_params.init_x_range(1), sim_params.init_x_range(2));
fprintf('  Init Y range: [%.1f, %.1f] cm\n', sim_params.init_y_range(1), sim_params.init_y_range(2));
fprintf('  Source position: (%.1f, %.1f) cm\n', sim_params.source_x, sim_params.source_y);

end
