function result = run_custom_plume_test(cfgFile)
%RUN_CUSTOM_PLUME_TEST Run the navigation model with a custom plume and log odor values.
%
%   RESULT = RUN_CUSTOM_PLUME_TEST(CFGFILE) runs the navigation model using
%   the configuration specified by CFGFILE. The script logs the odor(i,it)
%   values produced by the model. If CFGFILE is not provided, the default
%   configuration 'configs/my_complex_plume_config.yaml' is used.
%
%   Example:
%       run_custom_plume_test; % uses the default configuration
%       run_custom_plume_test('my_cfg.yaml');

if nargin < 1 || isempty(cfgFile)
    cfgFile = fullfile('configs','my_complex_plume_config.yaml');
end

% Ensure the project paths are set up
if isempty(which('run_navigation_cfg'))
    startup;
end

cfg = load_config(cfgFile);
result = run_navigation_cfg(cfg);

for it = 1:size(result.odor,2)
    for i = 1:size(result.odor,1)
        fprintf('odor(%d,%d) = %.4f\n', i, it, result.odor(i,it));
    end
end
end
