%%%%%%%%%%%%%%%%%%%%%%Data%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Server capacity: each location has its capacity
array[LOCATIONS] of int: capacity;
int: ub_server; int:lb_server;

%%%% array of facilities situation: 0-closed, 1-opened
int: Num_client;
set of int: CLIENTS = 1..Num_client;
int: Num_scenarios;
set of int: SCENARIO = 1..Num_scenarios;
array[SCENARIO,CLIENTS] of 0..1: client_array; %% scenario data

%%%% array of locations situation: 0-no using, 1-build on this position
int: Num_location;
set of int: LOCATIONS = 1..Num_location;
%%%% Revenue from facilities
array[CLIENTS,LOCATIONS] of int: revenue;
% Demand of client
array[CLIENTS,LOCATIONS] of int: demand;
% Cost of loaction
array[LOCATIONS] of int: cost;

%%%%%%%%%%%%%%%%%%%%Decision variables%%%%%%%%%%%%%%
array[LOCATIONS] of var 0..1: server; % which locations build server
array[SCENARIO,CLIENTS,LOCATIONS] of var 0..1: client; % clinet assign to which server

% server = [1,0,0,0,0];

array[LOCATIONS] of var 0..infinity: unserved = [0,0,0,0,0];

% array[LOCATIONS] of var 0..infinity : unserved;

%%%%%%%%%%%%%%%%%%%% Constraint %%%%%%%%%%%%%%%%%%%%%
%1. Total servers in range [lb,ub]
var int: num_fac= sum(i in LOCATIONS)(server[i]);
constraint num_fac>=lb_server /\ num_fac<=ub_server;

%2. Dont exceed loactions capacity
constraint forall(s in SCENARIO)
                (forall(l in LOCATIONS)
                (sum(f in CLIENTS)(demand[f,l]*client[s,f,l])<=capacity[l]*server[l]+unserved[l]));
                
%3. Don't assign closed clients to server, same for servers
constraint forall(s in SCENARIO)(forall(c in CLIENTS, l in LOCATIONS)(client_array[s,c]==0 -> client[s,c,l]==0));
constraint forall(s in SCENARIO)(forall(c in CLIENTS, l in LOCATIONS)(server[l]==0 -> client[s,c,l]==0));

%4. 
constraint forall(s in SCENARIO)(forall(c in CLIENTS)(client_array[s,c]==1 -> sum(l in LOCATIONS)(client[s,c,l])==1));

%%%%%%%%%%%%%%%%%%% Objective %%%%%%%%%%%%%%%%%%
% min : c*x_j - r_ij*y_ij
% array[LOCATIONS] of int: penalty; %penalty = 1000

var int: obj = sum(i in LOCATIONS)(cost[i]*server[i])*Num_scenarios-
               sum(s in SCENARIO)(sum(f in CLIENTS, l in LOCATIONS)(revenue[f,l]*client[s,f,l]));
               
solve minimize obj;

% output ["The objective is: "++show(obj)++"\n"
%         ++"The opened server is: "++show(server)++"\n"
%         ];
% output [show(client[c,l])
%         ++ if l=Num_location then "\n" else "," endif | 
%         c in CLIENTS, l in LOCATIONS];
% output ["Locate \(num_fac) servers with \(sum(c in CLIENTS, l in LOCATIONS where client[c,l]==1)(demand[c,l])) unit serve,\(unserve_c) unserved"];




%%%% data
%%%%% server server number
% Num_location=5;
% ub_server=5;
% lb_server=0;
% %%%%% client number
% Num_client=25;
% %% cost of location c_j
% cost=[40,60,47,68,60];
% %% revenue r_ij
% revenue=array2d(1..25, 1..5, [0, 22, 18, 14, 22, 15, 11, 20, 8, 14, 4, 22, 10, 0, 25, 14, 23, 23, 5, 22, 8, 23, 14, 5, 11, 18, 5, 2, 23, 6, 6, 8, 22, 3, 15, 14, 21, 6, 16, 14, 21, 6, 1, 8, 3, 16, 14, 13, 12, 22, 8, 20, 15, 15, 12, 11, 4, 9, 15, 11, 2, 19, 13, 2, 9, 15, 20, 17, 0, 16, 6, 1, 21, 23, 1, 11, 21, 2, 15, 17, 17, 17, 3, 13, 3, 15, 5, 14, 19, 7, 10, 8, 0, 8, 14, 22, 24, 23, 14, 15, 14, 13, 8, 2, 23, 21, 12, 10, 12, 17, 2, 10, 13, 10, 9, 20, 21, 9, 20, 21, 23, 18, 2, 9, 23]);
% %demand d_ij
% demand=array2d(1..25, 1..5, [0, 22, 18, 14, 22, 15, 11, 20, 8, 14, 4, 22, 10, 0, 25, 14, 23, 23, 5, 22, 8, 23, 14, 5, 11, 18, 5, 2, 23, 6, 6, 8, 22, 3, 15, 14, 21, 6, 16, 14, 21, 6, 1, 8, 3, 16, 14, 13, 12, 22, 8, 20, 15, 15, 12, 11, 4, 9, 15, 11, 2, 19, 13, 2, 9, 15, 20, 17, 0, 16, 6, 1, 21, 23, 1, 11, 21, 2, 15, 17, 17, 17, 3, 13, 3, 15, 5, 14, 19, 7, 10, 8, 0, 8, 14, 22, 24, 23, 14, 15, 14, 13, 8, 2, 23, 21, 12, 10, 12, 17, 2, 10, 13, 10, 9, 20, 21, 9, 20, 21, 23, 18, 2, 9, 23]);
% %capacity u_j
% capacity=[188,188,188,188,188];    
%penalty q_j0
% penalty = [1000,1000,1000,1000,1000];
%Num_scenarios
% Num_scenarios=50;

