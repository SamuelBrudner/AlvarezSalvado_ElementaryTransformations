

function out = navigation_model_vec(triallength, environment, plotting,ntrials)

% [x,y,heading,odor,successrate,latency,start,odorON,odorOFF]...
%     = navigation_model_vec(triallength, environment, plotting,ntrials)

% [x,y,heading,odor,odorON,odorOFF] = navigation_model_vec(triallength, environment, plotting,ntrials)
%
%   triallength -> Length of each trial in samples
%   environment -> String specifying the environment of the simulation.
%                  The options are:
%                        -'Crimaldi' uses real plume data from the GEN2
%                        set. Runs at 15 Hz.
%                        -'gaussian' uses a gaussian odor gradient with no
%                        wind. Runs at 50 Hz.
%                        - 'openlooppulse' has constant wind and provides a
%                        single, 10 s long square odor pulse in the middle
%                        of the trial (equivalent to the Nagel lab mini-wind
%                        tunnel paradigm). Runs at 50 Hz.
%                        - 'openlooppulsewb' is equivalent to the previous
%                        one but has no wind, so it will simulate
%                        wind-blind flies.
%                        - 'openlooppulse15' and 'openlooppulsewb15' are
%                        equivalent to the two previous ones but the model
%                        will run at 15 Hz instead of at 50 Hz.
%  plotting -> 1 to plot
%  ntrials -> number of trials (added by JDV)
%    if ntrials>1, x, y,heading, odor* are returned as arrays, with each
%    column corresponding to a separate trial.
%
% This program simulates one trial of a navigation model developed by the
% Nagel lab, based on walking Drosophila behavior. It generates
% trajectories and heading of a virtual fly in a virtual environment in
% response to odor signals.



tic
% Scaling factors
%tscale = 15/50; % Now loaded from config %15Hz/50Hz ratio to convert parameters for the plume data (parameters are expressed in samples at 50 Hz in this code)
%pxscale = 0.74; % Now loaded from config %mm/pixel ratio to convert pixels from the plume data to actual mm




%% MODEL PARAMETERS ---------------------------------------------------------
% Expressed in samples at 50 Hz


% ON/OFF MODEL PARAMETERS - uses compress-first models
 beta = 0.01;    % Kd for Hill compression
 tau_Aon = 490;   % adaptation time constant for ON (in samples at 50Hz)
 tau_Aoff = 504; % adaptation time constant for OFF (in samples at 50Hz)
 tau_ON = 36;   % time constant for ON filtering (in samples at 50 Hz, so ~ 1s)
 tau_OFF1 = 31;  % fast time constant for OFF differentiation (in samples at 50Hz)
 tau_OFF2 = 242; % slow time constant for OFF differentiation (in samples at 50Hz)
 
 scaleON = 1; % This parameter might be used to scale the ON response 
 scaleOFF = 1; % This parameter might be used to scale the OFF response 

 
 % NAVIGATION MODEL PARAMETERS
 turnbase = 0.12;  % baseline turn rate; this is a probability when used at 50Hz, we scale it when running at 15 Hz so the overall turn rate is the same
 tsigma = 20;     % sigma of turn amplitude distribution (degrees/s)
 vbase = 6;        % baseline velocity, mm/s

 tmodON =  0.03;    % strength of ON turning modulation 
 tmodOFF = 0.75;         % strength of OFF turning modulation 

 vmodON = 0.45;     % strength of ON velocity modulation (mm/s)
 vmodOFF = 0.8;    % strength of OFF velocity modulation (mm/s)

 kup = 5;          % strength of upwind drive (deg/samp)
 kdown = 0.5;       % strength of downwind drive (deg/samp)

 
 if (nargin<=3)
     ntrials=1;
 end
  switch environment % If we are using environments at 15 Hz, converts the time constants to 15 Hz
     
    case {'Crimaldi','crimaldi','openlooppulse15','openlooppulsewb15'}
        tau_Aon = tau_Aon*tscale;
        tau_Aoff = tau_Aoff*tscale;
        tau_ON = tau_ON*tscale;
        tau_OFF1 = tau_OFF1*tscale;
        tau_OFF2 = tau_OFF2*tscale;

        kup = kup/tscale;
        kdown = kdown/tscale;

        turnbase = turnbase/tscale;
        tmodON = tmodON/tscale;
        tmodOFF = tmodOFF/tscale;
  end

%allocate space for results
x = zeros(triallength,ntrials);
y = zeros(triallength,ntrials);
% heading = zeros(triallength,ntrials);
odor = zeros(triallength,ntrials);
odorON = zeros(triallength,ntrials);
odorOFF = zeros(triallength,ntrials);
downwind=zeros(triallength,ntrials);
upwind=zeros(triallength,ntrials);
  
%initial conditions
Aon = zeros(triallength,ntrials);
Aoff = zeros(triallength,ntrials);
C = zeros(triallength,ntrials);
Coff = zeros(triallength,ntrials);
R = zeros(triallength,ntrials);
Rh = zeros(triallength,ntrials);
Rd = zeros(triallength,ntrials);
ON = zeros(triallength,ntrials);
OFF = zeros(triallength,ntrials);
pturn = zeros(triallength,ntrials);
turn = zeros(triallength,ntrials);
N = zeros(triallength,ntrials);

% D function - to be scaled by the model below
D = -sin(2*pi*(1:360)/360);


%% INITIAL POSITION AND HEADING --------------------------------------------

% Determine initial position and heading
switch environment
    case {'Crimaldi', 'crimaldi'} % Real plume data from the Crimaldi lab
        % USE NEXT TWO LINES INSTEAD IF RUNNING MULTIPLE SIMULATIONS TO
        % COMPARE: It will use the same starting positions in every simulation
%         load('startpositions plume.mat');
%         x(1,:) = iniX(1:ntrials); y(1,:) = iniY(1:ntrials);
        % -----------------------------------------------------------------
        x(1,:) = (rand(1,ntrials).*16)-8; %rand(1,ntrials)*10-5; % gaussian distribution of initial X positions centered around 0, with sigma=4
        y(1,:) = rand(1,ntrials)*5-30; % random distribution of initial Y positions, between -25 and -30
    case {'openlooppulse15','openlooppulsewb15'} % Single odor pulses at 15 Hz
        x(1,:)= 0; y(1,:)= 0; triallength = 1050;
    case {'openlooppulse','openlooppulsewb','openloopslope'} % Single odor pulses at 50 Hz
        x(1,:) = 0; y(1,:) = 0; triallength = 3500;
    case {'gaussian', 'Gaussian'} % Gaussian odor gradient without any wind
        x(1,:) = 20*rand(1,ntrials)-10;
        y(1,:) = 20*rand(1,ntrials)-10;
end
heading = 360*rand(1,ntrials); % starts with a random heading




%% SIMULATE DIFFERENTIAL EQUATIONS -----------------------------------------


odormax = 1; % Maximal odor concentration for pulse environments
sigma = 5; % Width of gaussian gradient in cm
pmax = 1; % Maximal odor probability for gaussian environment
%plume_xlims=[1 216]; % Now from config
%plume_ylims=[1 406]; % Now from config

OLzero=zeros(3501,1);
OLodorlib.openlooppulse15.data    =OLzero; OLodorlib.openlooppulse15.data(450:600) = 1;
OLodorlib.openlooppulse15.ws=1;
OLodorlib.openlooppulse.data      =OLzero; OLodorlib.openlooppulse.data(1500:2000) = 1;
OLodorlib.openlooppulse.ws=1;
OLodorlib.openlooppulsewb15.data  =OLzero; OLodorlib.openlooppulsewb15.data(450:600) = 1;
OLodorlib.openlooppulsewb15.ws=0;
OLodorlib.openlooppulsewb.data    =OLzero; OLodorlib.openlooppulsewb.data(1500:2000)= 1;
OLodorlib.openlooppulsewb.ws=0;
OLodorlib.openloopslope.data    =[OLzero]; 
% OLodorlib.openloopslope.data(1501:2000) = 1; 
% OLodorlib.openloopslope.data(2001:3000) = 1/1000*fliplr([1:1000]);
OLodorlib.openloopslope.data = exp(-((([-1750:1750-1]).^2)/100000));
OLodorlib.openloopslope.ws=1;

ws=1;

for i = 1:triallength

    % Get odor concentration
    switch environment
        
        case {'Crimaldi', 'crimaldi'}
            %plume_filename = get_plume_file(); % Now loaded earlier from config
            tind = mod(i-1,3600)+1; % Restarts the count in case we want to run longer trials
            xind = round(10*x(i,:)/pxscale)+108; % turns the initial position to cm
            yind = -round(10*y(i,:)/pxscale)+1;
            out_of_plume=union(union(find(xind<plume_xlims(1)),find(xind>plume_xlims(2))),union(find(yind<plume_ylims(1)),find(yind>plume_ylims(2))));
            within=setdiff([1:ntrials],out_of_plume);
            odor(i,out_of_plume)=0;
            %this will be vectorizable if the dataset is loaded into memory
            for it=within
                odor(i,it)=max(0,h5read(plume_filename,dataset_name,[xind(it) yind(it) tind],[1 1 1])); % Draws odor concentration for the current position and time
            end
        case {'openloopslope','openlooppulse15','openlooppulse','openlooppulsewb15','openlooppulsewb'}
            odor(i,:) = odormax*OLodorlib.(environment).data(i);
            ws=OLodorlib.(environment).ws;
         case {'gaussian', 'Gaussian'}
            odor(i,:) = pmax*exp(-(x(i,:).^2+y(i,:).^2)/(2*sigma^2));
            ws = 0;
            p(i,:) = odor(i,:);
         
    end

    % Adaptation
    Aon(i+1,:) = Aon(i,:) + (odor(i,:) - Aon(i,:))/tau_Aon;
    Aoff(i+1,:) = Aoff(i,:) + (odor(i,:) - Aoff(i,:))/tau_Aoff;

    % ON Compression
    C(i,:) = odor(i,:)./(odor(i,:)+beta+Aon(i,:)); % compresses and adapts
    
    % ON response
    ON(i+1,:) = ON(i,:) + (C(i,:) - ON(i,:))/tau_ON; % filters
    odorON(i+1,:) = scaleON * ON(i+1,:);

    % OFF response (model 2, compress first)
    Coff(i+1,:) = odor(i,:)./(odor(i,:) + Aoff(i,:) + beta); % compression and adaptation
    R(i+1,:) = R(i,:) + (Coff(i,:) - R(i,:))/tau_OFF1; % fast filter
    Rh(i+1,:) = Rh(i,:) + (Coff(i,:) - Rh(i,:))/tau_OFF2; % slow filter
    Rd(i+1,:) = max(0,Rh(i,:)-R(i,:)); % difference of filtered vectors
    odorOFF(i+1,:) = scaleOFF * Rd(i+1,:);
    
 %%% alternate OFF response (model 1, filter first)
%     R(i+1,:) = R(i,:) + (odor(i,:) - R(i,:))/tau_OFF1; % fast filter
%     Rh(i+1,:) = Rh(i,:) + (odor(i,:) - Rh(i,:))/tau_OFF2; % slow filter
%     Rd(i+1,:) = max(0,Rh(i,:)-R(i,:)); % difference of filtered vectors
%     odorOFF(i+1,:) = scaleOFF * Rd(i,:)./(Rd(i,:) + Aoff(i,:) + beta);
 
 
    % Random turning
    pturn(i,:) = turnbase - tmodON*odorON(i,:) + tmodOFF*odorOFF(i,:); %generates probability of turning
    turn(i,:) = rand(1,ntrials)<pturn(i,:); % Checks if turning happens considering current probability
%     N(i,:) = turn(i,:).*(randn(1,ntrials).*tsigma); % Draws a random turn size from a gaussian distribution with a width of tsigma
    N(i,:) = turn(i,:).*tsigma.*(round(rand(1,ntrials))*2-1).*randn(1,ntrials).^2; % Draws a random turn size from a gaussian distribution with a width of tsigma

    
    % Wind-directed turning
    if (ws>0)
        H = round(mod(heading(i,:),359))+1; % Returns heading from 0 to 359
        downwind(i,:) = kdown*D(H); % Drive towards downwind
        upwind(i,:) = -kup*(D(H).*odorON(i,:)); % Drive towards upwind
    end

    % Computes final heading and velocity
    heading(i+1,:) = heading(i,:) + N(i,:) + downwind(i,:) + upwind(i,:); % sum turn signals to get heading
    v(i,:) = max(0, vbase*(1+vmodON*odorON(i,:)-vmodOFF*odorOFF(i,:)));

    % Calculate X and Y positions
    switch environment
        case {'Crimaldi','crimaldi','openlooppulse15','openlooppulsewb15'}
            [dx, dy] = pol2cart((heading(i,:)-90)/360*2*pi,v(i,:)/(10*plume_config.frame_rate));  % convert mm/s to cm/samp
        otherwise
           [dx, dy] = pol2cart((heading(i,:)-90)/360*2*pi,v(i,:)/500); % convert mm/s to cm/samp (for 50Hz)
    end
    x(i+1,:) = x(i,:) + dx;
    y(i+1,:) = y(i,:) + dy;
end

% A(end,:) = [];
Rh(end,:) = [];
ON(end,:) = [];
OFF(end,:) = [];
odorON(end,:) = [];
odorOFF(end,:) = [];

x(end,:) = []; y(end,:) = [];
heading(end,:) = [];
heading = heading-90;           % rotate h so upwind is 90 deg for display
t = (1:length(odorON))/50;


start = [x(1,:)', y(1,:)'];

%% SUCCESS RATE AND LATENCY
switch environment
    case {'Crimaldi','crimaldi','Gaussian','gaussian'} % only in certain environments

    
    angles = atan2(y,x);
    distances = y./sin(angles);
        
    for i = 1:ntrials                                                                          
        found = find(distances(:,i) <= 2,1);
%         found = find(pdist2([x(:,i),y(:,i)],[0,0])<=2,1);
        if ~isempty(found)
            success(i) = 1;
            switch environment
                case {'Crimaldi','crimaldi'}
                    latency(i) = found/plume_config.frame_rate; % Use config frame rate
                case {'gaussian','Gaussian'}
                    latency(i) = found/50;
            end
        elseif isempty(found)
            success(i) = 0;
            latency(i) = nan;
        end
    end
    successrate = sum(success)/ntrials; % Success rate
end


% Fills output structure
out.environment = environment;
out.x = x;
out.y = y;
out.theta = heading;
out.odor = odor;
out.start = start;
out.ON = odorON;
out.OFF = odorOFF;
out.turn = turn;
switch environment
    case {'Crimaldi','crimaldi','Gaussian','gaussian'}
        out.successrate = successrate;
        out.latency = latency;
    otherwise
        out.successrate = [];
        out.latency = [];
end


toc

%% PLOTTING ---------------------------------------------------------------
%multitrial plotting not implemented yet
if plotting>=1 % & (ntrials==1) % Plots the trajectory of the trial
    figure(1);
    set(gcf,'Position',[548 210 690 655]);

%     switch environment
%         case {'gaussian', 'Gaussian'}
%             load('gaussian.mat');
%             imagesc(-15:15,-15:15,gauss);
%     end

    hold on;
    
    % plot frame, source and success area
%     plot([-8 8 8 -8 -8],[-30 -30 0 0 -30],'k');
%     plot(0,0,'.k','markersize',15);
    viscircles([0,0],2,'color','g');
    
    % plot trajectories
    plot(x,y,'k','linewidth',2);
    xodor = x; yodor = y;
    xodor(find(odor<beta)) = nan;
    yodor(find(odor<beta)) = nan;

    xon = x; yon = y;
    xon(find(odorON<0.2)) = nan;
    yon(find(odorON<0.2)) = nan;

    xoff = x; yoff = y;
    xoff(find(odorOFF<0.1)) = nan;
    yoff(find(odorOFF<0.1)) = nan;
%     hold on; plot(xodor,yodor,'k');
    hold on; plot(xon,yon,'m','linewidth',2);
    hold on; plot(xoff,yoff,'c','linewidth',2);
    box off; set(gca,'TickDir','out');
    xlabel('x position (cm)');
    ylabel('y position (cm)');
%     title(['tmodON=',num2str(tmodON),' tmodOFF=',num2str(tmodOFF)]);
    axis equal;
%     xlim([-40 40]);
%     ylim([-65 2]);
    
    switch environment
        case {'Crimaldi', 'crimaldi'}
%             xlim([-12.5 12.5]); ylim([-36 5]);
%             plot(0,0,'.g','markersize',15);
        case {'openlooppulse','openlooppulsewb'}
            axis tight
    end

%     set(gcf,'Position',[5 318 558 460]);
%     xlim([-15 15]); ylim([-15 15]);
%     xlim([-10 10]); ylim([-10 10]);

end

if plotting>=2
    figure;
    set(gcf,'PaperPositionMode','auto');
    set(gcf,'Position',[44    50   329   613]);
    t = [1:length(odor)]/50;
    switch environment
        case {'Crimaldi','crimaldi','openlooppulse15','openlooppulsewb15'}
            t = (1:length(odor))/15;
    end
	color = 'k';

    subplot(5,1,1);
    hold on; plot(t,odor,color);
    box off; set(gca,'TickDir','out'); axis tight
    title('ODOR')

    subplot(5,1,2);
    hold off; plot(t,odorON,'m'); set(gca,'Ylim',[0 1]); ylabel('ON response');
    box off; set(gca,'TickDir','out'); axis tight
    title('ON')

    subplot(5,1,3);
    hold off; plot(t,odorOFF,'c'); set(gca,'Ylim',[0 1]); ylabel('OFF response');
    box off; set(gca,'TickDir','out'); axis tight
    title('OFF')

    subplot(5,1,4);
    %plot(t,mod(heading,360),color); ylabel('heading');
    plot(t,pturn,color); ylabel('Pturn');
    box off; set(gca,'TickDir','out'); axis tight
    title('P(TURN)')

    subplot(5,1,5);
    hold on; plot(t,v,color); ylabel('velocity');
    box off; set(gca,'TickDir','out'); axis tight
    title('GROUND SPEED')

    figure;
    set(gcf,'PaperPositionMode','auto');
    set(gcf,'Position',[375    50   329   613]);
    t = [1:length(odor)]/50;

    subplot(4,1,1);
    hold off; plot(t,turn,'k');
    box off; set(gca,'TickDir','out'); axis tight
    title('N')

    subplot(4,1,2);
    hold off; plot(t,upwind,'k'); %set(gca,'Ylim',[0 1]);
    box off; set(gca,'TickDir','out'); axis tight
    title('UPWIND')

    subplot(4,1,3);
    hold off; plot(t,downwind,'k');
    box off; set(gca,'TickDir','out'); axis tight
    title('DOWNWIND')

    subplot(4,1,4);
    hold off; plot(t,N + upwind + downwind,'k');
    box off; set(gca,'TickDir','out'); axis tight
    title('N + UPWIND + DOWNWIND')

%     figure('position',[182 725 317 260]);
%     plot([1:360]-180,-kup*D*50,'r');
%     hold on; plot([1:360]-180,kdown*D*50);



%     subplot(5,1,3); %hold off;
%   %  plot(t,o2,'b')
%    % hold on; plot(t,o1,'r');
%     hold on; plot(t,odor,'k');
% %    hold on; plot(t,oh1,'k:');
%    % hold on; plot(t,oh,'m');
%     set(gca,'Ylim',[0 1]);
%     box off; set(gca,'TickDir','out')

%     subplot(5,1,4);
%     hold on; plot(t,odorOFF,'k');
%     %hold on; plot(t,odorOFF2,'r');
%
%     subplot(5,1,5);
%     hold on; plot(t,odor,'k');

%     setX(0,max(t));
end

% if plotting>=3
%         % plot turn signals
%     figure;
%     set(gcf,'Position',[84    14   339   659]);
%     subplot(4,1,1); plot(t,50*dHdtupwind); box off; set(gca,'TickDir','out'); ylabel('upwind drive (deg/s)'); title('turn signals');
%     subplot(4,1,2); plot(t,50*dHdtcast); box off; set(gca,'TickDir','out'); ylabel('casting')
%     subplot(4,1,3); plot(t,50*dHdtdownwind); box off; set(gca,'TickDir','out'); ylabel('downwind drive')
%     subplot(4,1,4); plot(t,50*dHdtturn); box off; set(gca,'TickDir','out'); ylabel('turning')
% 
%     % plot heading, velocity, and position
%     figure;
%     set(gcf,'Position',[427    14   339   659]);
%     subplot(4,1,2); plot(t,heading); box off; set(gca,'TickDir','out'); ylabel('heading (degrees)'); title('heading signals');
%     subplot(4,1,3); plot(t,v); box off; set(gca,'TickDir','out'); ylabel('velocity (mm/s)')
%     subplot(4,1,1); plot(t,50*dHdt); box off; set(gca,'TickDir','out'); ylabel('total turn')
%     %subplot(4,1,4); plot(t,dx); box off; set(gca,'TickDir','out'); ylabel('x velocity (mm/s)')
% end

end

