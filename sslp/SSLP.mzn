%%%%%%%%%%%%%%%%%%%%%%Data%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Server capacity: each location has its capacity
array[LOCATIONS] of int: capacity;
int: ub_server; int:lb_server;

%%%% array of facilities situation: 0-closed, 1-opened
int: Num_client;
set of int: CLIENTS = 1..Num_client;
array[CLIENTS] of 0..1: client_array; %% scenario data

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
array[CLIENTS,LOCATIONS] of var 0..1: client; % clinet assign to which server


%%%%%%%%%%%%%%%%%%%% Constraint %%%%%%%%%%%%%%%%%%%%%
%1. Total servers in range [lb,ub]
var int: num_fac= sum(i in LOCATIONS)(server[i]);
constraint num_fac>=lb_server /\ num_fac<=ub_server;

%2. Dont exceed loactions capacity
constraint forall(l in LOCATIONS)
                (sum(f in CLIENTS)(demand[f,l]*client[f,l])<=capacity[l]*server[l]);
                
%3. Don't assign closed clients to server, same for servers
constraint forall(c in CLIENTS, l in LOCATIONS)(client_array[c]==0 -> client[c,l]==0);
constraint forall(c in CLIENTS, l in LOCATIONS)(server[l]==0 -> client[c,l]==0);

%4. 
constraint forall(c in CLIENTS)(client_array[c]==1 -> sum(l in LOCATIONS)(client[c,l])==1);

%%%%%%%%%%%%%%%%%%% Objective %%%%%%%%%%%%%%%%%%
% min : c*x_j - r_ij*y_ij

var int: obj = sum(i in LOCATIONS)(cost[i]*server[i])-
               sum(f in CLIENTS, l in LOCATIONS)(revenue[f,l]*client[f,l]);
               
var int: unserve_c = sum(c in CLIENTS)(client_array[c])-sum(c in CLIENTS, l in LOCATIONS)(client[c,l]);
solve minimize obj;

% output ["The objective is: "++show(obj)++"\n"
%         ++"The opened server is: "++show(server)++"\n"
%         ];
% output [show(client[c,l])
%         ++ if l=Num_location then "\n" else "," endif | 
%         c in CLIENTS, l in LOCATIONS];
% output ["Locate \(num_fac) servers with \(sum(c in CLIENTS, l in LOCATIONS where client[c,l]==1)(demand[c,l])) unit serve,\(unserve_c) unserved"];




% %%%% data
