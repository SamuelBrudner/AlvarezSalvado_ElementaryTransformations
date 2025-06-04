% diagnose_model_state.m - Check if the model is ready to run

fprintf('=== Model State Diagnostic ===\n\n');

all_good = true;

%% 1. Check configs
fprintf('1. Checking config files...\n');
configs = {
    'configs/plumes/crimaldi_10cms_bounded.json'
    'configs/plumes/smoke_1a_backgroundsubtracted.json'
};

for i = 1:length(configs)
    if exist(configs{i}, 'file')
        cfg = jsondecode(fileread(configs{i}));
        [~, name, ~] = fileparts(configs{i});
        
        has_model_params = isfield(cfg, 'model_params');
        has_tscale = has_model_params && isfield(cfg.model_params, 'tscale');
        has_pxscale = has_model_params && isfield(cfg.model_params, 'pxscale');
        
        if has_tscale && has_pxscale
            fprintf('   ✓ %s: tscale=%.3f, pxscale=%.3f\n', ...
                    name, cfg.model_params.tscale, cfg.model_params.pxscale);
        else
            fprintf('   ✗ %s: Missing model_params\n', name);
            all_good = false;
        end
    else
        fprintf('   ✗ Config not found: %s\n', configs{i});
        all_good = false;
    end
end

%% 2. Check get_plume_file
fprintf('\n2. Checking get_plume_file.m...\n');
try
    [file, config] = get_plume_file();
    if isfield(config, 'tscale') && isfield(config, 'pxscale')
        fprintf('   ✓ Returns tscale=%.3f, pxscale=%.3f\n', ...
                config.tscale, config.pxscale);
    else
        fprintf('   ✗ Does not return tscale/pxscale\n');
        all_good = false;
    end
catch ME
    fprintf('   ✗ Error: %s\n', ME.message);
    all_good = false;
end

%% 3. Check navigation model
fprintf('\n3. Checking navigation model...\n');
model_file = 'Code/Elifenavmodel_bilateral.m';
if exist(model_file, 'file')
    code = fileread(model_file);
    
    % Check for initialization code
    if contains(code, 'Initialize scaling factors')
        fprintf('   ✓ Has scaling factor initialization\n');
    else
        fprintf('   ✗ Missing scaling factor initialization\n');
        all_good = false;
    end
    
    % Check for config loading
    if contains(code, 'Load from config if running Crimaldi')
        fprintf('   ✓ Has config loading code\n');
    else
        fprintf('   ? May not load from config\n');
    end
else
    fprintf('   ✗ Model file not found!\n');
    all_good = false;
end

%% 4. Summary
fprintf('\n=== SUMMARY ===\n');
if all_good
    fprintf('✓ Everything looks good! Model should run without tscale errors.\n');
    fprintf('\nYou can run:\n');
    fprintf('  >> quick_test_both_plumes\n');
else
    fprintf('✗ Issues found. Run the fix:\n');
    fprintf('  >> complete_fix_workflow\n');
end