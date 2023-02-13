clear
clc
%% Parameters
num_sc = 12; % number of small cells
num_time_slot = 1008; % number of time slots

% load Milan Data set
R = csvread('C:\Users\yanglianrui\Desktop\milan_data.csv');

% primary traffic 
R = R(1:1008,1:13);

% secondary traffic
S =  1- (R(:,2:end));

%backup values
S_bkp = S;
R_bkp = R;

%power model parameters, ma macrocell  mi microcell pi picocell fe femocell
%rh remote radio head
p_mao = 130; k_ma = 4.7; p_matx = 20;
p_mio = 56;  k_mi = 2.6; p_mitx = 6.3;  P_mis = 39.0;
p_pio = 6.8; k_pi = 4.0; p_pitx = 0.13; P_pis = 4.3;
p_feo = 4.8; k_fe = 8.0; p_fetx = 0.05; P_fes = 2.9;
p_rho = 84;  k_rh = 2.8; p_rhtx = 20;   P_rhs = 56;  

% additional parameters
Ntx_ma = 1;
Ntx_sc = 1;
C_RB = 0.5;
N_RB_rh = 75; %15MHz
N_RB_mi = 50; %10MHz
N_RB_pi = 25; %5MHz
N_RB_fe = 15; %3MHz

% all possible switching pattern for each time slot
x = dec2bin(0:(2^num_sc -1)) - '0';

%initialize values
P_ON = zeros(num_time_slot, 1);
S_opt = zeros(num_time_slot,1);
T_opt = zeros(num_time_slot,1);
P_t = zeros(num_time_slot,2^num_sc);
Rev_t = zeros(num_time_slot,2^num_sc);

% TOTAL POWER CONSUMPTION WHEN ALL SCs ARE ON
for k = 1:num_time_slot
    P_ma_1 = Ntx_ma*(p_mao + k_ma*R_bkp(k,1)*p_matx);
    
    for n = 1:num_sc
        % for remote radio head (RRH)
        if n <= round(num_sc/4)
            P_sc_1(:,n) = Ntx_sc*(p_rho + k_rh*R_bkp(k, n+1)*p_rhtx);
        % for micro cell(mi)
        elseif (n > round(num_sc/4) && n <= round(2*num_sc/4))
            P_sc_1(:,n) = Ntx_sc*(p_mio + k_mi*R_bkp(k, n+1)*p_mitx);
        % for pico cell (pi) 
        elseif (n > round(2*num_sc/4) && n <= round(3*num_sc/4))
            P_sc_1(:,n) = Ntx_sc*(p_pio + k_pi*R_bkp(k, n+1)*p_pitx);
        % for femto cell (fe)    
        elseif (n > round(3*num_sc/4) && n <= round(4*num_sc/4))
            P_sc_1(:,n) = Ntx_sc*(p_feo + k_fe*R_bkp(k, n+1)*p_fetx);
        end
        
        P_ON(k) = P_ma_1  + sum(P_sc_1,2);
    end
end
 
% POWER CONSUMPTION DUE TO SWITCHING AND REVENUE DUE TO LEASING
for j = 1:num_time_slot
    l_max = 1; % maximum normilized capacity of the macro cell is set to 1, as the traffoc load of all cells are btw 0 and 1
    
    % back up values
    R2 = R(j,2:4);
    R3 = R(j,5:7);
    R4 = R(j,8:10);
    R5 = R(j,11:13);
    
    for i = 1:2^num_sc
        % Checking to see that the maximum normlized capacity of the macro
        % cell is not exceeded
        l_ma = R(j,1) + sum(R2.*((1 - x(i,1:3)))).*0.75 + (sum(R3.*(1 - x(i,4:6)))).*0.5 +...
            sum(R4.*((1 - x(i,7:9)))).*0.25 + sum(R5.*((1 - x(i,10:12)))).*0.15;
       
        if l_ma > l_max
            P_t(j,i) = inf;
            Rev_t(j,i) = -inf;
            
            continue;
        else
            % update traffic load of small cells after traffic offloading
            S(j,1:end) = S_bkp(j,1:end).*(1 - x(i,:));
            R(j,2:end) = R_bkp(j,2:end).*x(i,:);
            
            % MACRO POWER CONSUMPTION
            P_ma = Ntx_ma*(p_mao + k_ma*l_ma*p_matx);
            
            for sc_idx = 1:num_sc
                %Rev_sc(:,sc_idx) =  S(j,sc_idx)*C_RB*N_RB;
              
                % RRH POWER CONSUMPTION AND LEASING REVENUE
                if sc_idx <= round(num_sc/4)
                    Rev_sc(:,sc_idx) =  S(j,sc_idx)*C_RB*N_RB_rh;
                    if R(j,sc_idx+1) == 0
                        P_sc(:,sc_idx) = Ntx_sc*P_rhs;
                    else
                        P_sc(:,sc_idx) = Ntx_sc*(p_rho + k_rh*R(j, sc_idx+1)*p_rhtx);
                    end
                end
                
                % MICRO POWER CONSUMPTION AND LEASING REVENUE
                if (sc_idx > round(num_sc/4) && sc_idx <= round(2*num_sc/4))
                    Rev_sc(:,sc_idx) =  S(j,sc_idx)*C_RB*N_RB_mi;
                    if R(j,sc_idx+1) == 0
                        P_sc(:,sc_idx) = Ntx_sc*P_mis;
                    else
                        P_sc(:,sc_idx) = Ntx_sc*(p_mio + k_mi*R(j, sc_idx+1)*p_mitx);
                    end
                end
                % PICO POWER CONSUMPTION AND LEASING REVENUE
                if (sc_idx > round(2*num_sc/4) && sc_idx <= round(3*num_sc/4))
                    Rev_sc(:,sc_idx) =  S(j,sc_idx)*C_RB*N_RB_pi;
                    if R(j,sc_idx+1) == 0
                        P_sc(:,sc_idx) = Ntx_sc*P_pis;
                    else
                        P_sc(:,sc_idx) = Ntx_sc*(p_pio + k_pi*R(j, sc_idx+1)*p_pitx);
                    end
                end
                
                % FEMTO POWER CONSUMPTION AND LEASING REVENUE
                if (sc_idx > round(3*num_sc/4) && sc_idx <= round(4*num_sc/4))
                    Rev_sc(:,sc_idx) =  S(j,sc_idx)*C_RB*N_RB_fe;
                    if R(j,sc_idx+1) == 0
                        P_sc(:,sc_idx) = Ntx_sc*P_fes;
                    else
                        P_sc(:,sc_idx) = Ntx_sc*(p_feo + k_fe*R(j, sc_idx+1)*p_fetx);
                    end
                end
            end
            % Total power consumption per time slot for each switching
            % combination
            P_t(j,i) = P_ma  + sum(P_sc, 2);
            
           % Total revenue per time slot for each switching combination
            Rev_t(j,i) = sum(Rev_sc,2);
        end
    end
end
%%
% total power saving per time slot for each switching combination
% we have assimed here that for each unit of power saved is equivalent to
% $1 savings, we would put the actual value based on electricity unit cost
% later
P_save = P_ON - P_t;

% total revenue per time slot for each switching combination
J = P_save + Rev_t;

% T_opt is the maximum revenue per time slot while S_opt is the optimal
% switching pattern
[T_opt, S_opt] = max(J, [], 2);
Total=[T_opt, S_opt]
%%
T = T_opt;
S_opt_bin = x(S_opt,:); % optimal switching pattern that gives maximum revenue in binary
S_opt_dec = bi2de(S_opt_bin); % optimal switching pattern that gives maximum revenus in decimal
Rev_Max_dataset = [R S_bkp S_opt_bin S_opt_dec T]; % saving the results

%%
%csvwrite('C:\Users\2309848A\Dropbox\Figs\VFA\RevMax_12Scs_ES.csv', Rev_Max_dataset)
