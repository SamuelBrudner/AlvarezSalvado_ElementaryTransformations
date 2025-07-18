function plot_arena_elements(arena_bounds, init_x, init_y, success_radius, xlim_vals, ylim_vals, scale_cm)
%PLOT_ARENA_ELEMENTS Standard decorations for plume-navigation figures.
%
%   plot_arena_elements(arena_bounds, init_x, init_y, success_radius,
%                       xlim_vals, ylim_vals, scale_cm)
%
% Inputs
%   arena_bounds  – struct with fields x_min, x_max, y_min, y_max (cm)
%   init_x        – 1×2 vector, [x_min, x_max] of initialisation zone (cm)
%   init_y        – 1×2 vector, [y_min, y_max] of initialisation zone (cm)
%   success_radius– scalar, radius of the success circle (cm)
%   xlim_vals     – 1×2 vector for x-axis limits
%   ylim_vals     – 1×2 vector for y-axis limits
%   scale_cm      – scalar, length of the scale bar (cm)
%
% This helper centralises the repeated drawing of arena boundaries,
% start-zone rectangle, source circle/star, and scale bar so that the
% various plotting scripts stay consistent.  Modify here once and all
% plots update.
% Arena: black box, linewidth=2; Source: filled red circle; Grayscale plume backgrounds (set in scripts)
%

arguments
    arena_bounds struct
    init_x (1,2) double
    init_y (1,2) double
    success_radius (1,1) double {mustBePositive}
    xlim_vals (1,2) double
    ylim_vals (1,2) double
    scale_cm (1,1) double {mustBePositive}
end

cfg = get_plot_defaults();
if nargin < 7 || isempty(scale_cm)
    scale_cm = cfg.scale_bar_length_cm;
end

hold on;

% Arena rectangle
rectangle('Position', [arena_bounds.x_min, arena_bounds.y_min, ...
                       arena_bounds.x_max - arena_bounds.x_min, ...
                       arena_bounds.y_max - arena_bounds.y_min], ...
          'EdgeColor', cfg.colors.arena_edge, 'LineWidth', cfg.line_widths.arena_edge);
% Draw as a box (no fill)

% Initialisation zone
rectangle('Position', [init_x(1), init_y(1), diff(init_x), diff(init_y)], ...
          'EdgeColor', cfg.colors.init_zone_edge, 'LineWidth', cfg.line_widths.init_zone_edge);

% Success circle & source marker
th = linspace(0, 2*pi, 200);
plot(success_radius*cos(th), success_radius*sin(th), 'Color', cfg.colors.success_circle, ...
     'LineWidth', cfg.line_widths.success_circle);
% Draw source as filled red circle
scatter(0, 0, 80, [1 0 0], 'filled'); % 80 is marker area, [1 0 0] is red RGB

% Scale bar (bottom-right corner)
sbx = xlim_vals(2) - scale_cm - 1;
% place slightly above bottom limit
sby = ylim_vals(1) + 1;
plot([sbx, sbx + scale_cm], [sby, sby], 'Color', cfg.colors.scale_bar, ...
     'LineWidth', cfg.line_widths.scale_bar);
text(sbx + scale_cm/2, sby - 0.5, sprintf('%g cm', scale_cm), ...
     'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');

% Axes housekeeping
axis equal;
xlim(xlim_vals);
ylim(ylim_vals);

end
